import telebot
from time import time

"""Using request to send telebot msg"""
# import requests
# from json import dumps
# url2 = f'https://api.telegram.org/bot7922640117:AAG5b_YUjXHAVAGJxbMLth3heTEZwn442CA/sendMessage?chat_id=-1001417639887&parse_mode=HTML&text={msg}'
# headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
# requests.get(url2, headers=headers)


def telegram_msg(msg):
    bot = telebot.TeleBot("7922640117:AAG5b_YUjXHAVAGJxbMLth3heTEZwn442CA")
    response = bot.send_message("-4582896967", msg, parse_mode="HTML")
    return response


def telegram_photo():
    bot = telebot.TeleBot("7922640117:AAG5b_YUjXHAVAGJxbMLth3heTEZwn442CA")
    response = bot.send_photo(
        "-4582896967",
        f"https://alternative.me/crypto/fear-and-greed-index.png?a={time()}",
    )
    return response


if __name__ == "__main__":
    x = """
<html>
<body>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>symbol</th>
      <th>avg_hz</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>trumpusdt</td>
      <td>3.28</td>
    </tr>
    <tr>
      <td>stptusdt</td>
      <td>2.85</td>
    </tr>
    <tr>
      <td>quickusdt</td>
      <td>2.47</td>
    </tr>
    <tr>
      <td>peopleusdt</td>
      <td>2.37</td>
    </tr>
    <tr>
      <td>pnutusdt</td>
      <td>1.92</td>
    </tr>
    <tr>
      <td>agldusdt</td>
      <td>1.45</td>
    </tr>
    <tr>
      <td>bigtimeusdt</td>
      <td>1.39</td>
    </tr>
    <tr>
      <td>kernelusdt</td>
      <td>1.32</td>
    </tr>
    <tr>
      <td>eosusdt</td>
      <td>1.31</td>
    </tr>
    <tr>
      <td>bananas31usdt</td>
      <td>1.10</td>
    </tr>
    <tr>
      <td>meusdt</td>
      <td>1.09</td>
    </tr>
    <tr>
      <td>tstusdt</td>
      <td>1.01</td>
    </tr>
  </tbody>
</table>
</body>
</html>
"""
    telegram_msg(x)
    # telegram_photo()
