from players.base_player import Player
from openai import OpenAI
import time
from utils.conversation import get_conv_template


class OpenAIPlayer(Player):
    def __init__(self, public_name, delta=1, player_id=0, **kwargs):
        super().__init__(public_name, delta, player_id)
        assert 'model_name' in kwargs, 'model_name must be provided'
        self.model_name = kwargs['model_name']
        self.conv = get_conv_template("default")
        self.user_name = "user"
        self.client = OpenAI(api_key=kwargs.get('api_key'))

    def add_message(self, message, role='user'):
        if role == 'system':
            self.conv.system_message = message
        else:
            self.conv.append_message(self.conv.roles[0], message)

    def clean_conv(self):
        self.conv.messages = list()

    def get_text_answer(self, format_checker, decision=False):
        count = 0
        while count < 7:
            try:
                kwargs = {}
                if self.timeout:
                    kwargs['timeout'] = self.timeout

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=self.conv.to_openai_api_messages(),
                    **kwargs
                )
                self.text_response = response.choices[0].message.content.strip()
                if format_checker(self.text_response):
                    print(self.text_response)
                    self.conv.append_message(self.conv.roles[1], self.text_response)
                    break
            except Exception as e:
                if self.timeout and "timeout" in str(e).lower():
                    raise e
                print(e)
                time.sleep(4 ** (count + 1))
            count += 1
        return self.text_response

    def set_system_message(self, message):
        self.conv.system_message = message
