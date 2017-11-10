import sys
from PIL import Image
sys.path.insert(0, '/Workspace-Github/fruit_classification/code')
from telegram.ext import Updater, CommandHandler,MessageHandler, Filters
import R1_response as generate_response

TOKEN = '457979074:AAEovqh56cdBGtRxMx3YX8icpXULRMSQJ00' #whichfruit_bot
path = '/Workspace-Github/fruit_classification/test.jpg'

def start(bot, update):
    
    update.message.reply_text('Hello' + update.message.from_user.first_name + '!')

def reply(bot, update):
    
    print('Telegram Receive msg !!!')
    file_id = update.message.photo[-1]['file_id']
    photo = bot.getFile(file_id)
    photo.download(path)
    response = generate_response.respond()
    print('response: ', response)
    bot.send_message(chat_id=update.message.chat_id, text=response)
   
def reply2(bot, update):
    
    bot.send_message(chat_id=update.message.chat_id, text='Please send me a photo!')
    
# declaring handlers
updater = Updater(TOKEN)
#message_handler = MessageHandler(Filters.text, reply2)
message_handler = MessageHandler(Filters.all, reply)
start_handler = CommandHandler('start', start)

# adding handlers
updater.dispatcher.add_handler(message_handler)
updater.dispatcher.add_handler(start_handler)

# listen to requests
print('Ready to use!')
updater.start_polling()

