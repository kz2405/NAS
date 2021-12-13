import math
import numpy as np
from operator import itemgetter
#import cnn
import netparser
import state_enumerator as se

class StateStringUtils:
    ''' Contains all functions dealing with converting nets to net strings
        and net strings to state lists.
    '''
    def __init__(self, state_space_parameters):
        #self.image_size = state_space_parameters.image_size
        self.output_number = state_space_parameters.output_states
        self.enum = se.StateEnumerator(state_space_parameters)
    
    def add_drop_out_states(self, state_list):
        ''' Add drop out every 2 layers and after each fully connected layer
        Sets dropout rate to be between 0 and 0.5 at a linear rate
        '''
        new_state_list = []
        number_fc = len([state for state in state_list if state.layer_type == 'fc'])
        number_gap = len([state for state in state_list if state.layer_type == 'gap'])
        number_drop_layers = (len(state_list) - number_gap - number_fc)/2 + number_fc
        drop_number = 1
        for i in range(len(state_list)):
            new_state_list.append(state_list[i])
            if ((((i+1) % 2 == 0 and i != 0) or state_list[i].layer_type == 'fc')
                and state_list[i].terminate != 1
                and state_list[i].layer_type != 'gap'
                and drop_number <= number_drop_layers):
                drop_state = state_list[i].copy()
                drop_state.filter_depth = drop_number
                drop_state.fc_size = number_drop_layers
                drop_state.layer_type = 'dropout'
                drop_number += 1
                new_state_list.append(drop_state)

        return new_state_list

    def remove_drop_out_states(self, state_list):
        new_state_list = []
        for state in state_list:
            if state.layer_type != 'dropout':
                new_state_list.append(state)
        return new_state_list


    def state_list_to_string(self, state_list):
        '''Convert the list of strings to a string we can train from according to the grammar'''
        out_string = ''
        strings = []
        i = 0
        while i < len(state_list):
            state = state_list[i]
            if self.state_to_string(state):
                strings.append(self.state_to_string(state))
            i += 1
        return str('[' + ', '.join(strings) + ']')

    def state_to_string(self, state):
        ''' Returns the string asociated with state.
        '''
        if state.terminate == 1:
            return 'FC(%i, %s)' % (self.output_number, state.activation)
        # elif state.layer_type == 'conv':
        #     return 'C(%i,%i,%i)' % (state.filter_depth, state.filter_size, state.stride)
        # elif state.layer_type == 'gap':
        #     return 'GAP(%i)' % (self.output_number)
        # elif state.layer_type == 'pool':
        #     return 'P(%i,%i)' % (state.filter_size, state.stride)
        elif state.layer_type == 'lstm':
            return 'LSTM(%i,%s)' % (state.units, state.activation)
        elif state.layer_type == 'bilstm':
            return 'BILSTM(%i,%s)' % (state.units, state.activation)
        elif state.layer_type == 'rnn':
            return 'RNN(%i,%s)' % (state.units, state.activation)
        elif state.layer_type == 'gru':
            return 'GRU(%i,%s)' % (state.units, state.activation)
        elif state.layer_type == 'fc':
            return 'FC(%i, %s)' % (state.fc_size, state.activation)
        elif state.layer_type == 'dropout':
            return 'D(%f)' % (state.proba) ##SUPER BAD i am using fc_size and filter depth -- should fix later
        return None

    def convert_model_string_to_states(self, parsed_list, start_state=None):
        '''Takes a parsed model string and returns a recursive list of states.'''

        states = [start_state] if start_state else [se.State('start', 0, 0, None, 0, 0, 0,0, 0)]
            
        fc_occur = 0
        
        
        for layer in parsed_list:
            if layer[0] == 'lstm':
                states.append(se.State(layer_type='lstm',
                                    layer_depth=states[-1].layer_depth + 1,
                                    # filter_depth=layer[1],
                                    # filter_size=layer[2],
                                    # stride=layer[3],
                                    # image_size=states[-1].image_size,
                                    units=layer[1],
                                    activation=layer[2],
                                    proba=0,
                                    fc_depth=states[-1].fc_depth,
                                    fc_size=0,
                                    fc_occured = fc_occur,
                                    terminate=0))
            if layer[0] == 'bilstm':
                states.append(se.State(layer_type='bilstm',
                                    layer_depth=states[-1].layer_depth + 2,
                                    # filter_depth=layer[1],
                                    # filter_size=layer[2],
                                    # stride=layer[3],
                                    # image_size=states[-1].image_size,
                                    units=layer[1],
                                    activation=layer[2],
                                    proba=0,
                                    fc_depth=states[-1].fc_depth,
                                    fc_size=0,
                                    fc_occured = fc_occur,
                                    terminate=0))
            elif layer[0] == 'rnn':
                states.append(se.State(layer_type='rnn',
                                    layer_depth=states[-1].layer_depth + 1,
                                    # filter_depth=layer[1],
                                    # filter_size=layer[2],
                                    # stride=layer[3],
                                    # image_size=states[-1].image_size,
                                    units=layer[1],
                                    activation=layer[2],
                                    proba=0,
                                    fc_depth=states[-1].fc_depth,
                                    fc_size=0,
                                    fc_occured = fc_occur,
                                    terminate=0))
            elif layer[0] == 'gru':
                states.append(se.State(layer_type='gru',
                                    layer_depth=states[-1].layer_depth + 1,
                                    # filter_depth=layer[1],
                                    # filter_size=layer[2],
                                    # stride=layer[3],
                                    # image_size=states[-1].image_size,
                                    units=layer[1],
                                    activation=layer[2],
                                    proba=0,
                                    fc_depth=states[-1].fc_depth,
                                    fc_size=0,
                                    fc_occured = fc_occur,
                                    terminate=0))
            # elif layer[0] == 'gap':
            #     states.append(se.State(layer_type='gap',
            #                             layer_depth=states[-1].layer_depth + 1,
            #                             filter_depth=0,
            #                             filter_size=0,
            #                             stride=0,
            #                             image_size=1,
            #                             fc_size=0,
            #                             terminate=0))
            # elif layer[0] == 'pool':
            #     states.append(se.State(layer_type='pool',
            #                         layer_depth=states[-1].layer_depth + 1,
            #                         filter_depth=0,
            #                         filter_size=layer[1],
            #                         stride=layer[2],
            #                         image_size=self.enum._calc_new_image_size(states[-1].image_size, layer[1], layer[2]),
            #                         fc_size=0,
            #                         terminate=0))
            elif layer[0] == 'fc':
                fc_occur = 1
                if layer == parsed_list[-1]:
                    states.append(se.State(layer_type='fc',
                                    layer_depth=states[-1].layer_depth + 1,
                                    # filter_depth=len([state for state in states if state.layer_type == 'fc']),
                                    # filter_size=0,
                                    # stride=0,
                                    # image_size=0,
                                    units=0,
                                    activation=layer[2],
                                    proba=0,
                                    fc_depth=states[-1].fc_depth + 1,
                                    fc_size=layer[1],
                                    fc_occured = fc_occur,
                                    terminate=1))
                else:
                    states.append(se.State(layer_type='fc',
                                    layer_depth=states[-1].layer_depth + 1,
                                    # filter_depth=len([state for state in states if state.layer_type == 'fc']),
                                    # filter_size=0,
                                    # stride=0,
                                    # image_size=0,
                                    units=0,
                                    activation=layer[2],
                                    proba=0,
                                    fc_depth=states[-1].fc_depth + 1,
                                    fc_size=layer[1],
                                    fc_occured = fc_occur,
                                    terminate=0))
            elif layer[0] == 'dropout':
                states.append(se.State(layer_type='dropout',
                                        layer_depth=states[-1].layer_depth + 1,
                                        # filter_depth=layer[1],
                                        # filter_size=0,
                                        # stride=0,
                                        # image_size=states[-1].image_size,
                                        units=0,
                                    activation=None,
                                    proba=layer[1],
                                    fc_depth=states[-1].fc_depth,
                                    fc_size=0,
                                    fc_occured = fc_occur,
                                    terminate=0))

        return states



