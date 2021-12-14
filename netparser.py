from tensorflow import keras
import re
from yapps import runtime
from models.AutoML.state_space_parameters import possible_actvf

class NetScanner(runtime.Scanner):
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
        #('CONV', re.compile('C')),
        # ('POOL', re.compile('P')),
        ('SPLIT', re.compile('S')),
        ('LSTM', re.compile('LSTM')),
        ('BILSTM', re.compile('BILSTM')),
        ('RNN', re.compile('RNN')),
        ('GRU', re.compile('GRU')),
        ('FC', re.compile('FC')),
        ('DROP', re.compile('D')),
        # ('GLOBALAVE', re.compile('GAP')),
        # ('NIN', re.compile('NIN')),
        # ('BATCHNORM', re.compile('BN')),
        ('TERMINATE', re.compile('TERMINATE')),
        ('ACTIV', re.compile('|'.join(possible_actvf)))
    ]
    def __init__(self, str,*args,**kw):
        runtime.Scanner.__init__(self,None,{'\\s+':None,},str,*args,**kw)
        
class NetGenerating(runtime.Parser):
    Context = runtime.Context
    def layers(self, _parent=None):
        _context = self.Context(_parent, self._scanner, 'layers', [])
        _token = self._peek('LSTM', 'BILSTM', 'RNN', 'GRU', 'FC', 'DROP', 'TERMINATE', 'SPLIT', context=_context)     
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
        #if _token == 'CONV':
        #    conv = self.conv(_context)
        #    return conv
        #elif _token == 'NIN':
        #    nin = self.nin(_context)
        #    return nin
        #elif _token == 'GLOBALAVE':
        #    gap = self.gap(_context)
        #    return gap
        #elif _token == 'BATCHNORM':
        #    bn = self.bn(_context)
        #    return bn
        #elif _token == 'POOL':
        #    pool = self.pool(_context)
        #    return pool
        #elif _token == 'SPLIT':
        #    split = self.split(_context)
        #    return split
        elif _token == 'FC':
            fc = self.fc(_context)
            return fc
        elif _token == 'DROP':
            drop = self.drop(_context)
            return drop
        else: 
            terminate = self.terminate(_context)
            return terminate
#    def conv(self, _parent=None):
#        _context = self.Context(_parent, self._scanner, 'conv', [])
#        CONV = self._scan('CONV', context=_context)
#        result = ['conv']
#        numlist = self.numlist(_context)
#        return result + numlist

#    def nin(self, _parent=None):
#        _context = self.Context(_parent, self._scanner, 'nin', [])
#        NIN = self._scan('NIN', context=_context)
#        result = ['nin']
#        numlist = self.numlist(_context)
#        return result + numlist

#    def gap(self, _parent=None):
#        _context = self.Context(_parent, self._scanner, 'gap', [])
#        GLOBALAVE = self._scan('GLOBALAVE', context=_context)
#        result = ['gap']
#        numlist = self.numlist(_context)
#        return result + numlist

#    def bn(self, _parent=None):
#        _context = self.Context(_parent, self._scanner, 'bn', [])
#        BATCHNORM = self._scan('BATCHNORM', context=_context)
#        return ['bn']

#    def pool(self, _parent=None):
#        _context = self.Context(_parent, self._scanner, 'pool', [])
#        POOL = self._scan('POOL', context=_context)
#        result = ['pool']
#        numlist = self.numlist(_context)
#        return result + numlist
    
    def lstm(self, _parent=None):
        _context = self.Context(_parent, self._scanner, 'lstm', [])
        LSTM = self._scan('LSTM', context=_context)
        result = ['lstm']
        numlist = self.numlist(_context)
        return result + numlist
    
    def bilstm(self, _parent=None):
        _context = self.Context(_parent, self._scanner, 'bilstm', [])
        BiLSTM = self._scan('BILSTM', context=_context)
        result = ['bilstm']
        numlist = self.numlist(_context)
        return result + numlist
    
    def rnn(self, _parent=None):
        _context = self.Context(_parent, self._scanner, 'rnn', [])
        RNN = self._scan('RNN', context=_context)
        result = ['rnn']
        numlist = self.numlist(_context)
        return result + numlist
    
    def gru(self, _parent=None):
        _context = self.Context(_parent, self._scanner, 'gru', [])
        LSTM = self._scan('GRU', context=_context)
        result = ['gru']
        numlist = self.numlist(_context)
        return result + numlist
    
    def fc(self, _parent=None):
        _context = self.Context(_parent, self._scanner, 'fc', [])
        FC = self._scan('FC', context=_context)
        result = ['fc']
        numlist = self.numlist(_context)
        return result + numlist

    def drop(self, _parent=None):
        _context = self.Context(_parent, self._scanner, 'drop', [])
        DROP = self._scan('DROP', context=_context)
        result = ['dropout']
        numlist = self.numlist(_context)
        return result + numlist

    def terminate(self, _parent=None):
        _context = self.Context(_parent, self._scanner, 'terminate', [])
        TERMINATE = self._scan('TERMINATE', context=_context)
        result = ['terminate']
        return result 

    def split(self, _parent=None):
        _context = self.Context(_parent, self._scanner, 'split', [])
        SPLIT = self._scan('SPLIT', context=_context)
        self._scan('"\\\\{"', context=_context)
        result = ['split']
        net = self.net(_context)
        result.append(net)
        while self._peek('"\\\\}"', '","', context=_context) == '","':
            self._scan('","', context=_context)
            net = self.net(_context)
            result.append(net)
        self._scan('"\\\\}"', context=_context)
        return result

    def numlist(self, _parent=None):
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
    P = NetGenerating(NetScanner(text))
    return runtime.wrap_error_reporter(P, rule)



def caffe_to_keras(layer, rs = False):
    if layer[0] == "lstm":
        return keras.layers.LSTM(int(layer[1]), activation = layer[2], return_sequences = rs)
    if layer[0] == "bilstm":
        lstm = keras.layers.LSTM(int(layer[1]), activation = layer[2], return_sequences = rs)
        return keras.layers.Bidirectional(lstm)
    if layer[0] == "rnn":
        return keras.layers.SimpleRNN(int(layer[1]), activation = layer[2], return_sequences = rs)
    if layer[0] == "gru":
        return keras.layers.GRU(int(layer[1]), activation = layer[2], return_sequences = rs)
    #if layer[0] == 'conv':
    #    if first:
    #        return keras.layers.Conv2D(
    #        layer[1],
    #        layer[2],
    #        strides = (layer[3], layer[3]),
    #    )
    #    else:
    #        return keras.layers.Conv2D(
    #        layer[1],
    #        layer[2],
    #        strides = (layer[3], layer[3]),
    #        input_shape = (28,28,1)
    #    )
    elif layer[0] == "dropout":
        return keras.layers.Dropout(layer[1])
    elif layer[0] == "fc":
        return keras.layers.Dense(
        units=layer[1], activation = layer[2])
    elif layer[0] == "terminate":
         return keras.layers.Dense(units = 1, activation = "linear")
     
    #elif layer[0]== "pool":
    #    return keras.layers.MaxPooling2D(
    #    pool_size = (layer[1], layer[1]), 
    #    strides = layer[2],
    #    )
    #elif layer[0] == "gap":
    #    return keras.layers.GlobalAveragePooling2D()
    else:
        raise Exception

def parse_network_structure(net):
    keras.backend.clear_session()

    structure = []
    rs = False
    for layer_dict in net[::-1]:
        new_layer =caffe_to_keras(layer_dict, rs)
        if layer_dict[0] in ['lstm', 'bilstm', 'rnn','gru']:
            rs = True
        structure.append(new_layer)
    return structure[::-1]