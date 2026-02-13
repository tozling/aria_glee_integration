"""Bargaining environment for ARIA training.

Adapted from GLEE (https://github.com/eilamshapira/GLEE) for use in ARIA.
Original GLEE paper: https://arxiv.org/abs/2410.05254

Game: Two players (Alice and Bob) alternately propose how to divide a fixed
sum of money. Each player has a discount factor (delta) that reduces the value
of money each round. A player can accept or reject the proposal.

The ARIA agent plays as Alice (player 1) against an LLM opponent (Bob),
or vice versa, depending on agent_role.

Interface matches ARIA's existing environment convention:
  reset(config_idx=None) -> list[dict]  (initial message)
  _step(question=action) -> (next_obs, opponent_response, reward, done)
"""

import json
import re
import random
from typing import Optional, List, Dict, Tuple

from llm_base import llm_openai, vllm


# ─── Prompt templates (adapted from GLEE) ─────────────────────────────────────

RULES_TEMPLATE = (
    "Let's play a game.\n"
    "In this game, there are two players. You are playing as {player_name}.\n"
    "You and {rival_name} need to divide ${money_to_divide} between you.\n"
    "In each round of the game, one of the players will make a proposal for division.\n"
    "The other player will decide whether to accept the offer and finish the game, "
    "or to proceed to the next round, in which they will be the one making the proposal.\n"
    "{inflation_message}\n"
    "{max_rounds_message}"
)

OFFER_FORMAT_TEMPLATE = (
    'Send your offer to divide ${money_to_divide} in the JSON format:\n'
    '{{{{"\\"{player_name_lower}_gain\\"": The part that you will receive, '
    '\\"{rival_name_lower}_gain\\"": The part that {rival_name} will receive}}}}'
)

ACCEPT_FORMAT = 'Answer with {{"decision": "accept"}} or {{"decision": "reject"}}'


# ─── Game configurations ───────────────────────────────────────────────────────
# 48 configurations matching ARIA paper setup.
# Format: (delta_1, delta_2, money_to_divide, max_rounds, complete_information)
BARGAINING_CONFIGS: List[Tuple] = []
for _d1 in [0.8, 0.9]:
    for _d2 in [0.8, 0.9]:
        for _m in [100, 10000]:
            for _r in [6, 12]:
                for _ci in [True, False]:
                    BARGAINING_CONFIGS.append((_d1, _d2, _m, _r, _ci))
# 2 × 2 × 2 × 2 × 3 = 48 configurations


# ─── Utilities ────────────────────────────────────────────────────────────────

def _pretty(x: float) -> str:
    """Format a number without unnecessary trailing zeros."""
    s = f"{x:,.2f}"
    s = s.rstrip("0").rstrip(".")
    return s


def _lower_name(name: str) -> str:
    return name.lower().replace(" ", "_")


# ─── Main environment class ────────────────────────────────────────────────────

class BargainingEnv:
    """ARIA-compatible Bargaining environment.

    The agent plays as Alice or Bob against an LLM opponent.

    Odd rounds : Alice proposes, Bob decides.
    Even rounds: Bob proposes, Alice decides.

    Each call to _step() corresponds to ONE agent action (either making an
    offer or accepting/rejecting one). Intermediate steps return reward=0.0;
    the terminal reward is the agent's normalised discounted share.
    """

    def __init__(
        self,
        opponent_model: str = "gpt-4o",
        money_to_divide: float = 100.0,
        delta_1: float = 0.9,
        delta_2: float = 0.9,
        max_rounds: int = 12,
        complete_information: bool = True,
        messages_allowed: bool = False,
        agent_role: str = "alice",
    ):
        self.opponent_model = opponent_model
        self.money_to_divide = money_to_divide
        self.delta_1 = delta_1
        self.delta_2 = delta_2
        self.max_rounds = max_rounds
        self.complete_information = complete_information
        self.messages_allowed = messages_allowed
        self.agent_role = agent_role  # "alice" or "bob"

        # Derived player names
        self.agent_name = "Alice" if agent_role == "alice" else "Bob"
        self.opponent_name = "Bob" if agent_role == "alice" else "Alice"

        # Game state (initialised in reset)
        self.round_number: int = 1
        self.done: bool = True
        self.agent_history: List[Dict] = []
        self.opponent_history: List[Dict] = []
        self.agent_actions: List[str] = []
        self.opponent_actions: List[str] = []
        self._current_opponent_offer: Optional[Dict] = None
        self.game_result: Optional[Tuple] = None
        self.message: List[Dict] = []

    # ── Helpers ────────────────────────────────────────────────────────────────

    @property
    def _delta_agent(self) -> float:
        return self.delta_1 if self.agent_role == "alice" else self.delta_2

    @property
    def _delta_opponent(self) -> float:
        return self.delta_2 if self.agent_role == "alice" else self.delta_1

    def _is_agent_proposer(self, round_number: Optional[int] = None) -> bool:
        r = round_number if round_number is not None else self.round_number
        if self.agent_role == "alice":
            return r % 2 == 1   # Alice proposes in odd rounds
        else:
            return r % 2 == 0   # Bob proposes in even rounds

    def _inflation_msg(self, for_agent: bool = True) -> str:
        delta_self = self._delta_agent if for_agent else self._delta_opponent
        delta_other = self._delta_opponent if for_agent else self._delta_agent
        other_name = self.opponent_name if for_agent else self.agent_name

        loss_self = (1 - delta_self) * 100
        loss_other = (1 - delta_other) * 100

        if not loss_self and not loss_other:
            return ""
        prefix = (
            f"Beware of inflation! With each passing round, the money is worth "
            f"{_pretty(loss_self)}% less "
        )
        if self.complete_information:
            if loss_self != loss_other:
                prefix += (
                    f"for you. For {other_name}, the money is worth "
                    f"{_pretty(loss_other)}% less"
                )
        else:
            prefix += (
                f"for you. You don't know how inflation affects {other_name}"
            )
        return prefix.strip() + "."

    def _max_rounds_msg(self) -> str:
        if 0 < self.max_rounds <= 20:
            return (
                f"You have {self.max_rounds} rounds to divide the money, "
                "or both of you will get nothing!\n"
            )
        return ""

    def _rules_text(self, for_agent: bool = True) -> str:
        player = self.agent_name if for_agent else self.opponent_name
        rival = self.opponent_name if for_agent else self.agent_name
        return RULES_TEMPLATE.format(
            player_name=player,
            rival_name=rival,
            money_to_divide=_pretty(self.money_to_divide),
            inflation_message=self._inflation_msg(for_agent),
            max_rounds_message=self._max_rounds_msg(),
        )

    def _offer_format_text(self, for_agent: bool = True) -> str:
        player = self.agent_name if for_agent else self.opponent_name
        rival = self.opponent_name if for_agent else self.agent_name
        return (
            f"Send your offer to divide ${_pretty(self.money_to_divide)} "
            "in the JSON format:\n"
            f'{{"{_lower_name(player)}_gain": The part that you will receive, '
            f'"{_lower_name(rival)}_gain": The part that {rival} will receive}}'
        )

    def _inflation_update(self, round_number: int, for_agent: bool = True) -> str:
        delta = self._delta_agent if for_agent else self._delta_opponent
        loss = 1 - delta ** (round_number - 1)
        if not loss:
            return ""
        return (
            f"Due to inflation, the money you gain is worth "
            f"{_pretty(loss * 100)}% less than in the first round.\n"
        )

    # ── Prompt builders ────────────────────────────────────────────────────────

    def _proposer_prompt(self) -> str:
        inflation = self._inflation_update(self.round_number, for_agent=True)
        fmt = self._offer_format_text(for_agent=True)
        return (
            f"Round {self.round_number}\n"
            f"{inflation}"
            f"Send your offer to divide ${_pretty(self.money_to_divide)}.\n"
            f"{fmt}"
        )

    def _receiver_prompt(self, opp_offer: Dict) -> str:
        opp_key = f"{_lower_name(self.opponent_name)}_gain"
        agt_key = f"{_lower_name(self.agent_name)}_gain"
        opp_gain = opp_offer.get(opp_key, self.money_to_divide / 2)
        agt_gain = opp_offer.get(agt_key, self.money_to_divide / 2)
        inflation = self._inflation_update(self.round_number, for_agent=True)
        return (
            f"Round {self.round_number}\n"
            f"{inflation}"
            f"{self.opponent_name}'s offer:\n"
            f"# {self.agent_name} gain: ${_pretty(agt_gain)}\n"
            f"# {self.opponent_name} gain: ${_pretty(opp_gain)}\n"
            "Do you accept this offer?\n"
            '(Answer with {"decision": "accept"} or {"decision": "reject"})'
        )

    # ── Format validators ──────────────────────────────────────────────────────

    def _parse_offer(self, text: str) -> Tuple[bool, Optional[Dict]]:
        m = re.search(r"\{.*?\}", text, re.DOTALL)
        if not m:
            return False, None
        s = re.sub(r"(?<=\d),(?=\d{3})", "", m.group())
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            return False, None
        if not isinstance(obj, dict):
            return False, None
        agt_key = f"{_lower_name(self.agent_name)}_gain"
        opp_key = f"{_lower_name(self.opponent_name)}_gain"
        if agt_key not in obj or opp_key not in obj:
            return False, None
        if not isinstance(obj[agt_key], (int, float)):
            return False, None
        if not isinstance(obj[opp_key], (int, float)):
            return False, None
        if abs(obj[agt_key] + obj[opp_key] - self.money_to_divide) > 1.0:
            return False, None
        return True, obj

    @staticmethod
    def _parse_decision(text: str) -> Tuple[bool, Optional[str]]:
        m = re.search(r"\{.*?\}", text, re.DOTALL)
        if not m:
            return False, None
        try:
            obj = json.loads(m.group())
        except json.JSONDecodeError:
            return False, None
        if not isinstance(obj, dict) or "decision" not in obj:
            return False, None
        dec = obj["decision"]
        if dec not in ("accept", "reject"):
            return False, None
        return True, dec

    # ── LLM opponent helpers ───────────────────────────────────────────────────

    def _opponent_decide(self, agt_offer: Dict) -> Tuple[str, str]:
        """Ask opponent to accept/reject agent's offer. Returns (decision, raw_response)."""
        agt_key = f"{_lower_name(self.agent_name)}_gain"
        opp_key = f"{_lower_name(self.opponent_name)}_gain"
        agt_gain = agt_offer.get(agt_key, self.money_to_divide / 2)
        opp_gain = agt_offer.get(opp_key, self.money_to_divide / 2)
        inflation = self._inflation_update(self.round_number, for_agent=False)

        prompt_text = (
            f"Round {self.round_number}\n"
            f"{inflation}"
            f"{self.agent_name}'s offer:\n"
            f"# {self.opponent_name} gain: ${_pretty(opp_gain)}\n"
            f"# {self.agent_name} gain: ${_pretty(agt_gain)}\n"
            "Do you accept this offer?\n"
            '(Answer with {"decision": "accept"} or {"decision": "reject"})'
        )
        self.opponent_history.append({"role": "user", "content": prompt_text})
        raw = llm_openai(self.opponent_history, model=self.opponent_model) or ""
        if raw:
            self.opponent_history.append({"role": "assistant", "content": raw})
        self.opponent_actions.append(raw)

        ok, dec = self._parse_decision(raw)
        return (dec if ok else "reject"), raw

    def _opponent_propose(self) -> Tuple[Dict, str]:
        """Ask opponent to make an offer. Returns (parsed_offer, raw_response)."""
        opp_name = self.opponent_name
        agt_name = self.agent_name
        inflation = self._inflation_update(self.round_number, for_agent=False)
        prompt_text = (
            f"Round {self.round_number}\n"
            f"{inflation}"
            f"Send your offer to divide ${_pretty(self.money_to_divide)} "
            "in the JSON format:\n"
            f'{{"{_lower_name(opp_name)}_gain": The part that you will receive, '
            f'"{_lower_name(agt_name)}_gain": The part that {agt_name} will receive}}'
        )
        self.opponent_history.append({"role": "user", "content": prompt_text})
        raw = llm_openai(self.opponent_history, model=self.opponent_model) or ""
        if raw:
            self.opponent_history.append({"role": "assistant", "content": raw})
        self.opponent_actions.append(raw)

        m = re.search(r"\{.*?\}", raw, re.DOTALL)
        if m:
            s = re.sub(r"(?<=\d),(?=\d{3})", "", m.group())
            try:
                obj = json.loads(s)
                opp_key = f"{_lower_name(opp_name)}_gain"
                agt_key = f"{_lower_name(agt_name)}_gain"
                if opp_key in obj and agt_key in obj:
                    return obj, raw
            except json.JSONDecodeError:
                pass

        default = {
            f"{_lower_name(opp_name)}_gain": self.money_to_divide / 2,
            f"{_lower_name(agt_name)}_gain": self.money_to_divide / 2,
        }
        return default, raw

    # ── Reward ─────────────────────────────────────────────────────────────────

    def _reward(self, agent_gain: float, round_number: int) -> float:
        """Normalised discounted gain for the agent in [0, 1]."""
        return (self._delta_agent ** (round_number - 1)) * agent_gain / self.money_to_divide

    # ── Public interface ───────────────────────────────────────────────────────

    def reset(self, config_idx: Optional[int] = None) -> List[Dict]:
        """Reset game and return initial observation.

        Args:
            config_idx: Index into BARGAINING_CONFIGS (0-47). If None, uses
                        the parameters set in __init__.

        Returns:
            list: Initial message [{"role": "user", "content": "..."}]
        """
        if config_idx is not None and 0 <= config_idx < len(BARGAINING_CONFIGS):
            self.delta_1, self.delta_2, self.money_to_divide, self.max_rounds, \
                self.complete_information = BARGAINING_CONFIGS[config_idx]

        self.round_number = 1
        self.done = False
        self.agent_actions = []
        self.opponent_actions = []
        self._current_opponent_offer = None
        self.game_result = None

        self.agent_history = [{"role": "system", "content": self._rules_text(for_agent=True)}]
        self.opponent_history = [{"role": "system", "content": self._rules_text(for_agent=False)}]

        if self._is_agent_proposer():
            content = self._proposer_prompt()
        else:
            opp_offer, _ = self._opponent_propose()
            self._current_opponent_offer = opp_offer
            content = self._receiver_prompt(opp_offer)

        self.message = [{"role": "user", "content": content}]
        return self.message

    def _step(self, question: str) -> Tuple[List[Dict], str, float, bool]:
        """Process one agent action and advance game state.

        Args:
            question: Agent's action string (JSON offer or JSON decision).

        Returns:
            (next_obs, opponent_response, reward, done)
            next_obs         – list[dict] with the next prompt for the agent
            opponent_response – the opponent LLM's raw response string
            reward           – 0.0 for non-terminal steps; normalised gain at end
            done             – True when the game has ended
        """
        if self.done:
            return self.message, "", 0.0, True

        action = question.strip()
        self.agent_history.append({"role": "assistant", "content": action})
        self.agent_actions.append(action)

        if self._is_agent_proposer():
            return self._handle_agent_proposes(action)
        else:
            return self._handle_agent_decides(action)

    def _handle_agent_proposes(self, action: str) -> Tuple[List[Dict], str, float, bool]:
        ok, offer = self._parse_offer(action)
        if not ok:
            offer = {
                f"{_lower_name(self.agent_name)}_gain": self.money_to_divide / 2,
                f"{_lower_name(self.opponent_name)}_gain": self.money_to_divide / 2,
            }

        opp_decision, opp_raw = self._opponent_decide(offer)
        agt_key = f"{_lower_name(self.agent_name)}_gain"
        opp_key = f"{_lower_name(self.opponent_name)}_gain"
        agt_gain = offer.get(agt_key, self.money_to_divide / 2)

        if opp_decision == "accept":
            reward = self._reward(agt_gain, self.round_number)
            self.done = True
            self.game_result = (agt_gain, offer.get(opp_key, self.money_to_divide / 2), self.round_number)
            content = f"{self.opponent_name} accepted your offer! Game over."
            self.message = [{"role": "user", "content": content}]
            return self.message, opp_raw, reward, True

        # Rejected – advance round
        self.round_number += 1
        if self.round_number > self.max_rounds:
            self.done = True
            self.game_result = (0.0, 0.0, self.round_number - 1)
            content = (
                f"{self.opponent_name} rejected your offer. "
                f"No agreement reached after {self.max_rounds} rounds."
            )
            self.message = [{"role": "user", "content": content}]
            return self.message, opp_raw, 0.0, True

        # Opponent now proposes
        opp_offer, opp_offer_raw = self._opponent_propose()
        self._current_opponent_offer = opp_offer
        content = (
            f"{self.opponent_name} rejected your offer from round {self.round_number - 1}.\n"
            + self._receiver_prompt(opp_offer)
        )
        self.message = [{"role": "user", "content": content}]
        return self.message, opp_raw, 0.0, False

    def _handle_agent_decides(self, action: str) -> Tuple[List[Dict], str, float, bool]:
        ok, decision = self._parse_decision(action)
        if not ok:
            decision = "reject"

        opp_offer = self._current_opponent_offer or {}
        agt_key = f"{_lower_name(self.agent_name)}_gain"
        opp_key = f"{_lower_name(self.opponent_name)}_gain"
        agt_gain = opp_offer.get(agt_key, self.money_to_divide / 2)
        opp_gain = opp_offer.get(opp_key, self.money_to_divide / 2)

        if decision == "accept":
            reward = self._reward(agt_gain, self.round_number)
            self.done = True
            self.game_result = (agt_gain, opp_gain, self.round_number)
            content = f"You accepted {self.opponent_name}'s offer. Game over."
            self.message = [{"role": "user", "content": content}]
            return self.message, decision, reward, True

        # Rejected – advance round
        self.round_number += 1
        if self.round_number > self.max_rounds:
            self.done = True
            self.game_result = (0.0, 0.0, self.round_number - 1)
            content = (
                f"You rejected {self.opponent_name}'s offer. "
                f"No agreement reached after {self.max_rounds} rounds."
            )
            self.message = [{"role": "user", "content": content}]
            return self.message, decision, 0.0, True

        # Agent now proposes
        content = (
            f"You rejected {self.opponent_name}'s offer from round {self.round_number - 1}.\n"
            + self._proposer_prompt()
        )
        self.message = [{"role": "user", "content": content}]
        return self.message, decision, 0.0, False

    # ── Data export for reward aggregation ────────────────────────────────────

    def get_game_data(self) -> Dict:
        """Return game data in the format expected by reward_aggregation/bargaining_clustering/.

        Returns a dict with keys:
          alice_actions, bob_actions, alice_ids, bob_ids,
          game_result, config
        Embeddings (alice_action_embeddings, bob_actions_embeddings) are NOT
        included here; generate them separately with the OpenAI embeddings API.
        """
        if self.agent_role == "alice":
            alice_actions = self.agent_actions
            bob_actions = self.opponent_actions
        else:
            alice_actions = self.opponent_actions
            bob_actions = self.agent_actions

        alice_ids = [f"a_{i}" for i in range(len(alice_actions))]
        bob_ids = [f"b_{i}" for i in range(len(bob_actions))]

        return {
            "alice_actions": alice_actions,
            "bob_actions": bob_actions,
            "alice_ids": alice_ids,
            "bob_ids": bob_ids,
            "game_result": self.game_result,
            "config": {
                "delta_1": self.delta_1,
                "delta_2": self.delta_2,
                "money_to_divide": self.money_to_divide,
                "max_rounds": self.max_rounds,
                "complete_information": self.complete_information,
            },
        }
