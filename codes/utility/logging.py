import os
from datetime import datetime

class Logger():
    def __init__(self, filename, is_debug, path='D:\WorkSpace\CFMM\codes\logs1'):
        self.filename = filename
        self.path = path
        self.log_ = not is_debug
    def logging(self, s):
        s = str(s)
        print(datetime.now().strftime('%Y-%m-%d %H:%M: '), s)
        if self.log_:
            with open(os.path.join(os.path.join(self.path, 'baby')), 'a+') as f_log:  #, self.filename
                f_log.write(str(datetime.now().strftime('%Y-%m-%d %H:%M:  ')) + s + '\n')
