from models.AutoML.state_space_parameters import possible_actvf
from tensorflow import keras
import re
from yapps import runtime

'''This file holds code that parses architecture strings to turn them into Keras layers 
There are two classes: NetScanner which holds the regex patters, and NetParser which parses the strings'''

class NetScanner(runtime.Scanner):
    '''Scanner class that uses yapps runtime for regex on architecture strings'''
    patterns = [
        ('"\\\\]"', re.compile('\\]')),
        ('"\\\\["', re.compile('\\[')),
        ('"\\\\)"', re.compile('\\)')),
        ('"\\\\("', re.compile('\\(')),
        ('"\\\\}"', re.compile('\\}')),
        ('","', re.compile(',')),
        ('"\\\\{"', re.compile('\\{')),
        ('\\s+', re.compile('\\s+')),
        ('NUM', re.compile('[-+]?[0-9]*\.?[0-9]+')),
        ('LSTM', re.compile('LSTM')),
        ('BILSTM', re.compile('BILSTM')),
        ('RNN', re.compile('RNN')),
        ('GRU', re.compile('GRU')),
        ('FC', re.compile('FC')),
        ('DROP', re.compile('D')),
        ('TERMINATE', re.compile('TERMINATE')),
        ('ACTIV', re.compile('|'.join(possible_actvf)))
    ]
    def __init__(self, str,*args,**kw):
        runtime.Scanner.__init__(self,None,{'\\s+':None,},str,*args,**kw)
        
class NetParser(runtime.Parser):
    '''Parser class that calls yapps runtime scanner for regex on architecture strings'''
    Context = runtime.Context
    
    def layers(self, _parent=None):
        '''Parses a layer'''
        _context = self.Context(_parent, self._scanner, 'layers', [])
        _token = self._peek('LSTM', 'BILSTM', 'RNN', 'GRU', 'FC', 'DROP', 'TERMINATE', context=_context)     
        if _token == 'LSTM':
            ls = self.lstm(_context)
            return ls
        if _token == 'BILSTM':
            bils = self.bilstm(_context)
            return bils
        if _token == 'RNN':
            rnn = self.rnn(_context)
            return rnn
        if _token == 'GRU':
            gru = self.gru(_context)
            return gru
        elif _token == 'FC':
            fc = self.fc(_context)
            return fc
        elif _token == 'DROP':
            drop = self.drop(_context)
            return drop
        else: 
            terminate = self.terminate(_context)
            return terminate
    
    def lstm(self, _parent=None):
        '''Parses LSTM layer'''
        _context = self.Context(_parent, self._scanner, 'lstm', [])
        self._scan('LSTM', context=_context)
        result = ['lstm']
        numlist = self.numlist(_context)
        return result + numlist
    
    def bilstm(self, _parent=None):
        '''Parses biLSTM layer'''
        _context = self.Context(_parent, self._scanner, 'bilstm', [])
        self._scan('BILSTM', context=_context)
        result = ['bilstm']
        numlist = self.numlist(_context)
        return result + numlist
    
    def rnn(self, _parent=None):
        '''Parses RNN layer'''
        _context = self.Context(_parent, self._scanner, 'rnn', [])
        self._scan('RNN', context=_context)
        result = ['rnn']
        numlist = self.numlist(_context)
        return result + numlist
    
    def gru(self, _parent=None):
        '''Parses GRU layer'''
        _context = self.Context(_parent, self._scanner, 'gru', [])
        self._scan('GRU', context=_context)
        result = ['gru']
        numlist = self.numlist(_context)
        return result + numlist
    
    def fc(self, _parent=None):
        '''Parses FC layer'''
        _context = self.Context(_parent, self._scanner, 'fc', [])
        self._scan('FC', context=_context)
        result = ['fc']
        numlist = self.numlist(_context)
        return result + numlist

    def drop(self, _parent=None):
        '''Parses Dropout layer'''
        _context = self.Context(_parent, self._scanner, 'drop', [])
        self._scan('DROP', context=_context)
        result = ['dropout']
        numlist = self.numlist(_context)
        return result + numlist

    def terminate(self, _parent=None):
        _context = self.Context(_parent, self._scanner, 'terminate', [])
        self._scan('TERMINATE', context=_context)
        result = ['terminate']
        return result 

    def numlist(self, _parent=None):
        '''Parses the arguments (number of neurons and activation function) of a layer'''
        _context = self.Context(_parent, self._scanner, 'numlist', [])
        self._scan('"\\\\("', context=_context)
        result = []
        NUM = self._scan('NUM', context=_context)
        result.append(float(NUM))
        while self._peek('"\\\\)"', '","', context=_context) == '","':
            self._scan('","', context=_context)
            activ_fn = self._scan('ACTIV', context=_context)
            result.append(activ_fn)
        self._scan('"\\\\)"', context=_context)
        return result
    
    def net(self, _parent=None):
        ''' Parses the whole architecture string'''
        _context = self.Context(_parent, self._scanner, 'net', [])
        self._scan('"\\\\["', context=_context)
        result = []
        layers = self.layers(_context)
        result.append(layers)
        while self._peek('"\\\\]"', '","', context=_context) == '","':
            self._scan('","', context=_context)
            layers = self.layers(_context)
            result.append(layers)
        self._scan('"\\\\]"', context=_context)
        return result
    
def parse(rule, text):
    '''Parses an architecture string into a list representation '''
    P = NetParser(NetScanner(text))
    return runtime.wrap_error_reporter(P, rule)



def caffe_to_keras(layer, rs = False):
    ''' Turns a list representation of a layer into a Keras layer'''
    if layer[0] == "lstm":
        return keras.layers.LSTM(int(layer[1]), activation = layer[2], return_sequences = rs)
    if layer[0] == "bilstm":
        lstm = keras.layers.LSTM(int(layer[1]), activation = layer[2], return_sequences = rs)
        return keras.layers.Bidirectional(lstm)
    if layer[0] == "rnn":
        return keras.layers.SimpleRNN(int(layer[1]), activation = layer[2], return_sequences = rs)
    if layer[0] == "gru":
        return keras.layers.GRU(int(layer[1]), activation = layer[2], return_sequences = rs)
    elif layer[0] == "dropout":
        return keras.layers.Dropout(layer[1])
    elif layer[0] == "fc":
        return keras.layers.Dense(
        units=layer[1], activation = layer[2])
    elif layer[0] == "terminate":
         return keras.layers.Dense(units = 1, activation = "linear")
    else:
        raise Exception

def parse_network_structure(net):
    '''Turns a list representation of an architecture into a list of Keras layers '''
    keras.backend.clear_session()

    structure = []
    rs = False
    for layer_dict in net[::-1]:
        new_layer =caffe_to_keras(layer_dict, rs)
        if layer_dict[0] in ['lstm', 'bilstm', 'rnn','gru']:
            rs = True
        structure.append(new_layer)
    return structure[::-1]