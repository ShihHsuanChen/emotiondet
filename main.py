from emotiondet.settings import AppSettings
from emotiondet.app import create_app


try:
    settings = AppSettings()
except Exception as e:
    print(e)
    exit()
else:
    app = create_app(settings)
