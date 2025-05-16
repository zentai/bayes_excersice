# 文件路径：quanthunt/utils/telegram_bot.py

import os
import telebot
from quanthall.dispatcher import QuantHall
from quanthunt.hunterverse.interface import StrategyParam, Symbol

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
AUTHORIZED_USER_ID = int(os.getenv("TELEGRAM_USER_ID", "0"))

hall = QuantHall()
bot = telebot.TeleBot(TELEGRAM_TOKEN)


@bot.message_handler(commands=["start"])
def start(message):
    print("Your Telegram ID:", message.from_user.id)  # 👈 加这一行
    if message.from_user.id != AUTHORIZED_USER_ID:
        return
    bot.reply_to(
        message,
        "Hello, commander. Use /mission <symbol> <interval> <funds> <cap> or /summary",
    )


@bot.message_handler(commands=["summary"])
def summary(message):
    if message.from_user.id != AUTHORIZED_USER_ID:
        return
    mission = hall.scan_status()
    balance = hall.bank.get_balance()
    msg = f"💰 QuantHunt Summary\nActive Missions: {len(mission)}\nRemaining Funds: {balance:.2f}\n{mission}"
    bot.reply_to(message, msg)


@bot.message_handler(commands=["mission"])
def mission(message):
    if message.from_user.id != AUTHORIZED_USER_ID:
        return
    try:
        args = message.text.strip().split()[1:]
        symbol, interval, funds, cap = args[0], args[1], float(args[2]), float(args[3])
        strategy = StrategyParam(
            symbol=Symbol(symbol),
            interval=interval,
            funds=funds,
            stake_cap=cap,
            hmm_split=5,
            api_key="TBD",
            secret_key="TBD",
        )
        active = hall.scan_status()
        if hall.can_launch(strategy, active):
            hall.launch_task(strategy)
            bot.reply_to(message, f"Mission launched for {symbol} ({interval})")
        else:
            bot.reply_to(message, f"Mission for {symbol} cannot be launched.")
    except Exception as e:
        bot.reply_to(message, f"Error: {e}")


def telegram_msg(msg):
    response = bot.send_message(AUTHORIZED_USER_ID, msg, parse_mode="HTML")
    return response


def launch_bot():
    telegram_msg("Telegram bot is running...")
    bot.polling()


if __name__ == "__main__":
    launch_bot()
