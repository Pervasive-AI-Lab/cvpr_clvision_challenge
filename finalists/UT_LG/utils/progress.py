import os
import time


class WorkSplitter(object):
    def __init__(self):
        try:
            _, columns = os.popen('stty size', 'r').read().split()
            self.columns = int(columns)
        except:
            self.columns = 50

    def section(self, name):
        name_length = len(name)
        left_length = int((self.columns-name_length)/2)
        right_length = int(self.columns-name_length-left_length)

        output = '='*self.columns+'\n' \
                 + "|"+' '*(left_length-1)+name+' '*(right_length-1)+'|\n'\
                 + '='*self.columns+'\n'

        print(output)

    def subsection(self, name):
        name_length = len(name)
        left_length = int((self.columns-name_length)/2)
        right_length = int(self.columns-name_length-left_length)

        output = '#' * (left_length-1) + ' ' + name + ' ' + '#' * (right_length-1) + '\n'
        print(output)

    def subsubsection(self, name):
        name_length = len(name)
        left_length = int((self.columns-name_length)/2)
        right_length = self.columns-name_length-left_length

        output = '-' * (left_length-1) + ' ' + name + ' ' + '-' * (right_length-1) + '\n'
        print(output)


def inhour(elapsed):
    return time.strftime('%H:%M:%S', time.gmtime(elapsed))