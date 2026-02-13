"""Negotiation environment for ARIA training.

Adapted from GLEE (https://github.com/eilamshapira/GLEE) for use in ARIA.
Original GLEE paper: https://arxiv.org/abs/2410.05254

Game: A seller (Alice) and a buyer (Bob) negotiate the price of a product.
  - Seller's minimum value: V_A = M * F_A  (won't accept below this)
  - Buyer's maximum value:  V_B = M * F_B  (won't pay above this)
  - Seller's utility at agreed price p:  p - V_A
  - Buyer's utility  at agreed price p:  V_B - p
At any point, either player can end the negotiation by "dealing with John"
(outside option): the seller sells to John at V_A, the buyer buys from John
at V_B – both yielding 0 surplus.

The ARIA agent plays as Alice (seller) or Bob (buyer) against an LLM opponent.

Interface matches ARIA's existing environment convention:
  reset(config_idx=None) -> list[dict]  (initial message)
  _step(question=action) -> (next_obs, opponent_response, reward, done)
"""

import json
import re
from typing import Optional, List, Dict, Tuple

from llm_base import llm_openai, vllm


# ─── Prompt templates (adapted from GLEE) ─────────────────────────────────────

SELLER_RULES_TEMPLATE = (
    "You are {player_name}. You are selling one product that is worth no less "
    "than ${self_value} to you.\n"
    "{rival_name} is a potential buyer to whom you are offering the product. "
    "{complete_info_msg}\n"
    "You will offer {rival_name} to buy the product at a price of your choice. "
    "{rival_name} can either accept or reject the offer.\n"
    "Your goal is to earn as much money as you can for the product.\n"
    "{next_rounds_info}"
)

BUYER_RULES_TEMPLATE = (
    "You are playing as {player_name}. You are looking for a product that is "
    "worth no more than ${self_value} to you.\n"
    "{rival_name} is a seller trying to sell you the product. "
    "{complete_info_msg}\n"
    "{rival_name} will offer you a price to buy the product. You can either "
    "accept the offer or reject it.\n"
    "Your goal is to buy some product but save as much money as you can.\n"
    "{next_rounds_info}"
)


# ─── Game configurations ───────────────────────────────────────────────────────
# 48 configurations matching ARIA paper setup.
# Format: (f_a, f_b, price_scale, max_rounds, complete_information)
# seller_value = price_scale * f_a,  buyer_value = price_scale * f_b
# Only pairs where f_a <= f_b are included (deal is mutually beneficial).
NEGOTIATION_CONFIGS: List[Tuple] = []
_FA_FB_PAIRS = [
    (0.8, 1.0), (0.8, 1.2), (0.8, 1.5),
    (1.0, 1.2), (1.0, 1.5),
    (1.2, 1.5),
]  # 6 pairs
for _fa, _fb in _FA_FB_PAIRS:
    for _m in [100, 10000]:          # 2 scales
        for _r in [1, 10]:           # 2 round limits
            for _ci in [True, False]: # 2 info modes
                NEGOTIATION_CONFIGS.append((_fa, _fb, _m, _r, _ci))
# 6 × 2 × 2 × 2 = 48 configurations


# ─── Utilities ────────────────────────────────────────────────────────────────

def _pretty(x: float) -> str:
    s = f"{x:,.2f}"
    s = s.rstrip("0").rstrip(".")
    return s


# ─── Main environment class ────────────────────────────────────────────────────

class NegotiationEnv:
    """ARIA-compatible Negotiation environment.

    The agent plays as the seller (Alice) or buyer (Bob) against an LLM.

    Odd rounds : Seller (Alice) proposes price, Buyer (Bob) responds.
    Even rounds: Buyer proposes, Seller responds.

    Each call to _step() corresponds to ONE agent action (making a price offer
    or responding to one). Possible responses for the receiver:
      {"decision": "AcceptOffer"}   – accept the proposed price
      {"decision": "RejectOffer"}   – reject and (if rounds remain) counter-offer
      {"decision": "SellToJhon"}    – seller exits to outside option (utility 0)
      {"decision": "BuyFromJhon"}   – buyer exits to outside option (utility 0)

    Intermediate steps return reward=0.0; terminal reward is the agent's
    normalised surplus.
    """

    VALID_DECISIONS = ("AcceptOffer", "RejectOffer", "SellToJhon", "BuyFromJhon")

    def __init__(
        self,
        opponent_model: str = "gpt-4o",
        f_a: float = 1.0,
        f_b: float = 1.2,
        price_scale: float = 100.0,
        max_rounds: int = 10,
        complete_information: bool = True,
        messages_allowed: bool = False,
        agent_role: str = "alice",   # "alice" = seller, "bob" = buyer
    ):
        self.opponent_model = opponent_model
        self.f_a = f_a
        self.f_b = f_b
        self.price_scale = price_scale
        self.max_rounds = max_rounds
        self.complete_information = complete_information
        self.messages_allowed = messages_allowed
        self.agent_role = agent_role  # "alice" (seller) or "bob" (buyer)

        # Derived
        self.agent_name = "Alice" if agent_role == "alice" else "Bob"
        self.opponent_name = "Bob" if agent_role == "alice" else "Alice"
        self._is_seller = (agent_role == "alice")

        # Game state
        self.round_number: int = 1
        self.done: bool = True
        self.agent_history: List[Dict] = []
        self.opponent_history: List[Dict] = []
        self.agent_actions: List[str] = []
        self.opponent_actions: List[str] = []
        self._current_opponent_offer: Optional[float] = None
        self.game_result: Optional[Tuple] = None
        self.message: List[Dict] = []

    # ── Computed values ────────────────────────────────────────────────────────

    @property
    def seller_value(self) -> float:
        return self.price_scale * self.f_a

    @property
    def buyer_value(self) -> float:
        return self.price_scale * self.f_b

    @property
    def agent_value(self) -> float:
        return self.seller_value if self._is_seller else self.buyer_value

    @property
    def opponent_value(self) -> float:
        return self.buyer_value if self._is_seller else self.seller_value

    # ── Role helpers ───────────────────────────────────────────────────────────

    def _is_agent_proposer(self, round_number: Optional[int] = None) -> bool:
        r = round_number if round_number is not None else self.round_number
        if self.agent_role == "alice":   # seller
            return r % 2 == 1           # Alice proposes in odd rounds
        else:                            # buyer
            return r % 2 == 0           # Bob proposes in even rounds

    # ── Complete-information message ───────────────────────────────────────────

    def _complete_info_msg(self, for_agent: bool = True) -> str:
        other_name = self.opponent_name if for_agent else self.agent_name
        other_val = self.opponent_value if for_agent else self.agent_value
        if self.complete_information:
            return (
                f"The product is worth ${_pretty(other_val)} to {other_name}."
            )
        return f"You don't know the value of the product to {other_name}."

    # ── Next-rounds info ───────────────────────────────────────────────────────

    def _next_rounds_seller(self, seller_name: str, buyer_name: str,
                            seller_val: float) -> str:
        if self.max_rounds == 1:
            msg = (
                f"If {buyer_name} rejects the offer, you will sell the product "
                "to another buyer, John, "
            )
        else:
            msg = (
                f"If {buyer_name} rejects the offer, they can make a counteroffer. "
                "You can either accept or reject their counteroffer.\n"
            )
            if self.max_rounds <= 20:
                msg += f"You have {self.max_rounds} rounds to close the deal. However, "
            msg += (
                f"at any moment, you can choose to stop the negotiation with "
                f"{buyer_name} and sell the product to another buyer, John, "
            )
        msg += f"who is willing to buy the product from you for ${_pretty(seller_val)}."
        return msg

    def _next_rounds_buyer(self, seller_name: str, buyer_name: str,
                           buyer_val: float) -> str:
        if self.max_rounds == 1:
            msg = (
                "If you reject the offer, you will buy the product from "
                "another seller, John, "
            )
        else:
            msg = (
                f"If you reject the offer, you can make a counteroffer. "
                f"{seller_name} can either accept or reject your counteroffer.\n"
            )
            if self.max_rounds <= 20:
                msg += f"You have {self.max_rounds} rounds to close the deal. However, "
            msg += (
                "at any moment, you can choose to stop the negotiation and "
                "buy the product from another seller, John, "
            )
        msg += f"who is willing to sell the product to you for ${_pretty(buyer_val)}."
        return msg

    # ── Rules builders ─────────────────────────────────────────────────────────

    def _rules_text(self, for_agent: bool = True) -> str:
        if for_agent:
            player = self.agent_name
            rival = self.opponent_name
            self_val = self.agent_value
            is_seller = self._is_seller
        else:
            player = self.opponent_name
            rival = self.agent_name
            self_val = self.opponent_value
            is_seller = not self._is_seller

        ci_msg = self._complete_info_msg(for_agent)
        seller_name = player if is_seller else rival
        buyer_name = rival if is_seller else player

        if is_seller:
            next_info = self._next_rounds_seller(seller_name, buyer_name, self_val)
            return SELLER_RULES_TEMPLATE.format(
                player_name=player,
                rival_name=rival,
                self_value=_pretty(self_val),
                complete_info_msg=ci_msg,
                next_rounds_info=next_info,
            )
        else:
            next_info = self._next_rounds_buyer(seller_name, buyer_name, self_val)
            return BUYER_RULES_TEMPLATE.format(
                player_name=player,
                rival_name=rival,
                self_value=_pretty(self_val),
                complete_info_msg=ci_msg,
                next_rounds_info=next_info,
            )

    # ── Prompt builders ────────────────────────────────────────────────────────

    def _proposer_prompt(self, is_first: bool = False) -> str:
        offer_word = "offer" if is_first else "counteroffer"
        round_prefix = (
            f"Round {self.round_number}\n" if 1 < self.max_rounds <= 20 else ""
        )
        fmt = (
            f'Send your {offer_word} to {self.opponent_name}.\n'
            f'{{"product_price": The price you want to '
            f'{"sell" if self._is_seller else "buy"} the product for}}'
        )
        return f"{round_prefix}Send your {offer_word} to {self.opponent_name}.\n{fmt}"

    def _receiver_prompt(self, proposer_name: str, price: float) -> str:
        round_prefix = (
            f"Round {self.round_number}\n" if 1 < self.max_rounds <= 20 else ""
        )
        # Receiver options
        if self._is_seller:
            # Agent is seller receiving from buyer
            john_caption = "SellToJhon"
            john_text = f"Sell the product to John for ${_pretty(self.agent_value)}"
            act_name = "sell the product"
        else:
            # Agent is buyer receiving from seller
            john_caption = "BuyFromJhon"
            john_text = f"Buy the product from John for ${_pretty(self.agent_value)}"
            act_name = "buy the product"

        counteroffer_option = (
            f'(2) Reject {proposer_name}\'s offer and send a counteroffer\n'
            if self.round_number < self.max_rounds
            else f'(2) Reject {proposer_name}\'s offer\n'
        )
        accept_fmt = '{"decision": "AcceptOffer"}'
        reject_fmt = '{"decision": "RejectOffer"}'
        john_fmt = f'{{"decision": "{john_caption}"}}'

        return (
            f"{round_prefix}"
            f"{proposer_name}'s offer: The product price will be "
            f"${_pretty(price)}.\n"
            f"You have three options:\n"
            f"(1) Accept {proposer_name}'s offer, and {act_name} for "
            f"${_pretty(price)}\n"
            f"{counteroffer_option}"
            f"(3) {john_text}\n"
            f"Answer with {accept_fmt}, {reject_fmt}, or {john_fmt}"
        )

    # ── Format validators ──────────────────────────────────────────────────────

    @staticmethod
    def _parse_offer(text: str) -> Tuple[bool, Optional[float]]:
        m = re.search(r"\{.*?\}", text, re.DOTALL)
        if not m:
            return False, None
        s = m.group().replace("$", "")
        s = re.sub(r"(?<=\d),(?=\d{3})", "", s)
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            return False, None
        if not isinstance(obj, dict) or "product_price" not in obj:
            return False, None
        try:
            price = float(str(obj["product_price"]).replace("$", ""))
            return True, price
        except (ValueError, TypeError):
            return False, None

    def _parse_decision(self, text: str) -> Tuple[bool, Optional[str]]:
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
        if dec not in self.VALID_DECISIONS:
            return False, None
        return True, dec

    # ── LLM opponent helpers ───────────────────────────────────────────────────

    def _opponent_propose(self) -> Tuple[float, str]:
        """Ask opponent to propose a price. Returns (price, raw_response)."""
        opp_is_seller = not self._is_seller
        offer_word = "offer" if self.round_number == 1 else "counteroffer"
        round_prefix = (
            f"Round {self.round_number}\n" if 1 < self.max_rounds <= 20 else ""
        )
        verb = "sell" if opp_is_seller else "buy"
        prompt_text = (
            f"{round_prefix}Send your {offer_word} to {self.agent_name}.\n"
            f'{{"product_price": The price you want to {verb} the product for}}'
        )
        self.opponent_history.append({"role": "user", "content": prompt_text})
        raw = llm_openai(self.opponent_history, model=self.opponent_model) or ""
        if raw:
            self.opponent_history.append({"role": "assistant", "content": raw})
        self.opponent_actions.append(raw)

        ok, price = self._parse_offer(raw)
        if ok and price is not None:
            return price, raw
        # Default: midpoint
        default_price = (self.seller_value + self.buyer_value) / 2
        return default_price, raw

    def _opponent_decide(self, price: float) -> Tuple[str, str]:
        """Ask opponent to respond to agent's offer. Returns (decision, raw)."""
        opp_is_seller = not self._is_seller
        if opp_is_seller:
            john_caption = "SellToJhon"
            john_text = f"Sell to John for ${_pretty(self.opponent_value)}"
            act_name = "sell"
        else:
            john_caption = "BuyFromJhon"
            john_text = f"Buy from John for ${_pretty(self.opponent_value)}"
            act_name = "buy"

        counteroffer_option = (
            f'(2) Reject {self.agent_name}\'s offer and send a counteroffer\n'
            if self.round_number < self.max_rounds
            else f'(2) Reject {self.agent_name}\'s offer\n'
        )
        round_prefix = (
            f"Round {self.round_number}\n" if 1 < self.max_rounds <= 20 else ""
        )
        prompt_text = (
            f"{round_prefix}"
            f"{self.agent_name}'s offer: The product price will be "
            f"${_pretty(price)}.\n"
            f"You have three options:\n"
            f"(1) Accept {self.agent_name}'s offer, and {act_name} for "
            f"${_pretty(price)}\n"
            f"{counteroffer_option}"
            f"(3) {john_text}\n"
            f'Answer with {{"decision": "AcceptOffer"}}, {{"decision": "RejectOffer"}}, '
            f'or {{"decision": "{john_caption}"}}'
        )
        self.opponent_history.append({"role": "user", "content": prompt_text})
        raw = llm_openai(self.opponent_history, model=self.opponent_model) or ""
        if raw:
            self.opponent_history.append({"role": "assistant", "content": raw})
        self.opponent_actions.append(raw)

        ok, dec = self._parse_decision(raw)
        return (dec if ok else "RejectOffer"), raw

    # ── Reward ─────────────────────────────────────────────────────────────────

    def _reward(self, agreed_price: float) -> float:
        """Normalised surplus for the agent.

        Seller: (p - V_A) / (V_B - V_A)  clipped to [0, 1]
        Buyer:  (V_B - p) / (V_B - V_A)  clipped to [0, 1]
        If V_A >= V_B (no mutually beneficial deal exists), return 0.
        """
        surplus = self.buyer_value - self.seller_value
        if surplus <= 0:
            return 0.0
        if self._is_seller:
            gain = agreed_price - self.seller_value
        else:
            gain = self.buyer_value - agreed_price
        return max(0.0, min(1.0, gain / surplus))

    # ── Public interface ───────────────────────────────────────────────────────

    def reset(self, config_idx: Optional[int] = None) -> List[Dict]:
        """Reset game and return initial observation.

        Args:
            config_idx: Index into NEGOTIATION_CONFIGS (0-47). If None, uses
                        the parameters set in __init__.

        Returns:
            list: Initial message [{"role": "user", "content": "..."}]
        """
        if config_idx is not None and 0 <= config_idx < len(NEGOTIATION_CONFIGS):
            self.f_a, self.f_b, self.price_scale, self.max_rounds, \
                self.complete_information = NEGOTIATION_CONFIGS[config_idx]

        self.round_number = 1
        self.done = False
        self.agent_actions = []
        self.opponent_actions = []
        self._current_opponent_offer = None
        self.game_result = None

        self.agent_history = [{"role": "system", "content": self._rules_text(for_agent=True)}]
        self.opponent_history = [{"role": "system", "content": self._rules_text(for_agent=False)}]

        if self._is_agent_proposer():
            content = self._proposer_prompt(is_first=True)
        else:
            opp_price, _ = self._opponent_propose()
            self._current_opponent_offer = opp_price
            content = self._receiver_prompt(self.opponent_name, opp_price)

        self.message = [{"role": "user", "content": content}]
        return self.message

    def _step(self, question: str) -> Tuple[List[Dict], str, float, bool]:
        """Process one agent action and advance game state.

        Args:
            question: Agent's action string. Either:
                      - JSON price offer: {"product_price": <float>}
                      - JSON decision:    {"decision": "<AcceptOffer|RejectOffer|...>"}

        Returns:
            (next_obs, opponent_response, reward, done)
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
        ok, price = self._parse_offer(action)
        if not ok or price is None:
            price = (self.seller_value + self.buyer_value) / 2

        opp_decision, opp_raw = self._opponent_decide(price)

        if opp_decision == "AcceptOffer":
            reward = self._reward(price)
            self.done = True
            self.game_result = ("deal", price, self.round_number)
            content = f"{self.opponent_name} accepted your offer at ${_pretty(price)}. Game over."
            self.message = [{"role": "user", "content": content}]
            return self.message, opp_raw, reward, True

        if opp_decision in ("SellToJhon", "BuyFromJhon"):
            self.done = True
            self.game_result = ("john", price, self.round_number)
            content = (
                f"{self.opponent_name} rejected your offer and chose "
                "to deal with John. Game over."
            )
            self.message = [{"role": "user", "content": content}]
            return self.message, opp_raw, 0.0, True

        # RejectOffer – advance round
        self.round_number += 1
        if self.round_number > self.max_rounds:
            self.done = True
            self.game_result = ("no_deal", None, self.round_number - 1)
            content = (
                f"{self.opponent_name} rejected your offer. "
                f"No deal after {self.max_rounds} rounds."
            )
            self.message = [{"role": "user", "content": content}]
            return self.message, opp_raw, 0.0, True

        # Opponent now proposes
        opp_price, opp_offer_raw = self._opponent_propose()
        self._current_opponent_offer = opp_price
        content = (
            f"{self.opponent_name} rejected your offer from round {self.round_number - 1}.\n"
            + self._receiver_prompt(self.opponent_name, opp_price)
        )
        self.message = [{"role": "user", "content": content}]
        return self.message, opp_raw, 0.0, False

    def _handle_agent_decides(self, action: str) -> Tuple[List[Dict], str, float, bool]:
        ok, decision = self._parse_decision(action)
        if not ok:
            decision = "RejectOffer"

        price = self._current_opponent_offer
        if price is None:
            price = (self.seller_value + self.buyer_value) / 2

        if decision == "AcceptOffer":
            reward = self._reward(price)
            self.done = True
            self.game_result = ("deal", price, self.round_number)
            content = f"You accepted {self.opponent_name}'s offer at ${_pretty(price)}. Game over."
            self.message = [{"role": "user", "content": content}]
            return self.message, decision, reward, True

        if decision in ("SellToJhon", "BuyFromJhon"):
            self.done = True
            self.game_result = ("john", None, self.round_number)
            content = "You chose to deal with John. Game over."
            self.message = [{"role": "user", "content": content}]
            return self.message, decision, 0.0, True

        # RejectOffer – advance round
        self.round_number += 1
        if self.round_number > self.max_rounds:
            self.done = True
            self.game_result = ("no_deal", None, self.round_number - 1)
            content = (
                f"You rejected {self.opponent_name}'s offer. "
                f"No deal after {self.max_rounds} rounds."
            )
            self.message = [{"role": "user", "content": content}]
            return self.message, decision, 0.0, True

        # Agent now proposes
        content = (
            f"You rejected {self.opponent_name}'s offer from round {self.round_number - 1}.\n"
            + self._proposer_prompt(is_first=False)
        )
        self.message = [{"role": "user", "content": content}]
        return self.message, decision, 0.0, False

    # ── Data export for reward aggregation ────────────────────────────────────

    def get_game_data(self) -> Dict:
        """Return game data in the format expected by reward_aggregation/negotiation_clustering/.

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
                "f_a": self.f_a,
                "f_b": self.f_b,
                "price_scale": self.price_scale,
                "seller_value": self.seller_value,
                "buyer_value": self.buyer_value,
                "max_rounds": self.max_rounds,
                "complete_information": self.complete_information,
            },
        }
