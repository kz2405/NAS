import math
import netparser
import numpy as np


'''
This file holds the code for the layer transition algorithm.
It has two classes: State(), which represents any layer, and 
                    StateEnumerator(), which defines the action space given a current state, 
                    according to the following rules:
                     
1. Any non-termination state s can directly go to a termination state.
2. We only allow transitions from a non-termination, non-bilstm-layer-type state s with depth i to a state with layer depth i+1; 
   or for a bilstm-layer-type state s with depth i to a state with layer depth i+2, to ensure no loops in the state-action graph.
3. Any state that reaches the maximum layer depth may only transition to a termination state. 
4. We limit the number of fully connected (FC) layers in the whole network to be at maximum max_fc to keep the set of learnable parameters reasonably small.
   If more than one FC layer is allowed, then for any consecutive FC layers, the latter FC layer should not have more neurons than the FC layer before it.
5. We don’t allow two consecutive dropout layers nor a dropout layer to be the first layer.
6. Only RNN types of layers are candidates for the initial layer.
'''


class State:
    '''
    Defines a layer as a state

    '''
    def __init__(self,
                 layer_type=None,         
                 layer_depth=None,        
                 units=None,              
                 activation=None,         
                 proba=None,              
                 fc_depth=None,           
                 fc_size=None,            
                 fc_occured= None,        
                 terminate=None,          
                 state_list=None): 
        '''
        Constructor of State
        Parameters:
            layer_type: str e.g. 'lstm', 'dropout', 'fc', 'rnn', etc. -- layer type
            layer_depth: int -- current depth of network
            units: int -- number of neurons (Used for rnn, lstm, gru, bilstm; 0 otherwise)
            activation: str -- activation function (Used for all non-dropout layers)
            proba: float -- dropout ratio (Used for dropout; 0 otherwise)
            fc_depth: int -- current number of fc layers in the network
            fc_size: int -- number of fc neurons (Used for fc; 0 otherwise)
            fc_occured: bin -- whether the network contains any fc layer
            terminate: bin -- whether the state is a termination state
            state_list: list -- the state can be constructed from a list instead, list takes precedent

        Return:
            An instance of State (representing a layer)
        '''
        if not state_list: # if state_list is empty
            self.layer_type = layer_type
            self.layer_depth = layer_depth
            self.units = units
            self.activation = activation
            self.proba = proba
            self.fc_depth = fc_depth
            self.fc_size = fc_size
            self.fc_occured = fc_occured
            self.terminate = terminate
        else: #if not empty
            self.layer_type = state_list[0]
            self.layer_depth = state_list[1]
            self.units = state_list[2]
            self.activation = state_list[3]
            self.proba = state_list[4]  
            self.fc_depth = state_list[5]   
            self.fc_size = state_list[6]   
            self.fc_occured = state_list[7]
            self.terminate = state_list[8] 
        
    def as_tuple(self):
        '''
        Returns the state as a tuple.
        '''
        return (self.layer_type, 
                int(self.layer_depth), 
                int(self.units),
                self.activation,
                round(self.proba, 2),
                int(self.fc_depth),
                int(self.fc_size),
                int(self.fc_occured),
                int(self.terminate))
    
    def as_list(self):
        '''
        Returns the state as a list.
        '''
        return list(self.as_tuple())
    
    def copy(self):
        '''
        Returns a copy of the state.
        '''
        return State(self.layer_type, 
                     int(self.layer_depth), 
                      int(self.units),
                      self.activation,
                      round(self.proba, 2),
                      int(self.fc_depth),
                      int(self.fc_size),
                      int(self.fc_occured),
                      int(self.terminate))


class StateEnumerator:
    '''Class that deals with:
            Enumerating States (defining their possible transitions)

    '''
    def __init__(self, state_space_parameters):
        '''
        Constructor of StateEnumerator
        Parameters:
            state_space_parameters: state space parameters declared in state_space_parameters.py

        Returns an instance of StateEnumerator

        '''
        self.ssp = state_space_parameters
        self.layer_limit = state_space_parameters.layer_limit
        self.output_states = state_space_parameters.output_states 

    def enumerate_state(self, state, q_values):
        '''Defines all state transitions, populates q_values where actions are valid

           RNN types: LSTM, GRU, Bidirectional LSTM, RNN

            Legal Transitions:
            RNN         -> RNN, dropout     (IF state.layer_depth < layer_limit)
            RNN         -> fc               (If state.layer_depth < layer_limit and state.fc_depth < self.ssp.max_fc)
            RNN         -> terminate        (Always)

            dropout     -> RNN              (If state.layer_depth < layer_limit and not fc_occured)
            dropout     -> fc               (If state.layer_depth < layer_limit and state.fc_depth < self.ssp.max_fc)
            dropout     -> terminate        (Always)

            fc          -> fc               (If state.layer_depth < layer_limit AND state.fc_depth < self.ssp.max_fc) 
            fc          -> dropout          (IF state.layer_depth < layer_limit)
            fc          -> terminate        (Always)

            Updates and Returns: q_values 

        '''
        actions = []
        if state.layer_depth == 0: # initial layer
            for celltype in self.ssp.possible_celltypes:
                for unit in self.ssp.possible_units:
                    for activ_func in self.ssp.possible_actvf:
                            actions += [State(layer_type=celltype,
                                                layer_depth=state.layer_depth + 1,
                                                units = unit,
                                                activation = activ_func,
                                                proba = 0,
                                                fc_depth = state.fc_depth,
                                                fc_size=0,
                                                fc_occured = 0,
                                                terminate=0)]

        elif state.terminate == 0: # non-initial layer
            
            # First, add a terminate state (since the agent can choose to terminate from any non-termination state)
            # If we are at the layer limit, the only action left is to terminate
            actions += [State(layer_type='terminate',
                                   layer_depth=state.layer_depth + 1,
                                    units=0,
                                    activation='linear',
                                    proba=0,
                                    fc_depth=state.fc_depth + 1,
                                    fc_size=1,
                                    fc_occured = 1,
                                    terminate=1)]
            
            if state.layer_depth < self.layer_limit:
                # add RNN-types of layers
                if (state.layer_type in ['start', 'lstm', 'bilstm', 'rnn', 'gru', 'dropout']) and state.fc_occured == 0: 
                    for celltype in self.ssp.possible_celltypes:
                        for unit in self.ssp.possible_units:
                            for activ_func in self.ssp.possible_actvf:
                                actions += [State(layer_type=celltype,
                                                layer_depth=state.layer_depth + 1,
                                                units = unit,
                                                activation = activ_func,
                                                proba = 0,
                                                fc_depth = state.fc_depth,
                                                fc_size=0,
                                                fc_occured = state.fc_occured,
                                                terminate=0)]
                    if state.layer_depth + 1 < self.layer_limit:
                        for unit in self.ssp.possible_units:
                            for activ_func in self.ssp.possible_actvf:
                                actions += [State(layer_type='bilstm',
                                                layer_depth=state.layer_depth + 2,
                                                units = unit,
                                                activation = activ_func,
                                                proba = 0,
                                                fc_depth = state.fc_depth,
                                                fc_size=0,
                                                fc_occured = state.fc_occured,
                                                terminate=0)]

                # add dropout states -- iterate through all possible dropout rates 
                if (state.layer_type in ['lstm','bilstm', 'gru', 'rnn','fc'] or (state.layer_type == 'dropout' and self.ssp.allow_consecutive_dropout)): 
                    for rate in self.ssp.possible_proba:
                        actions += [State(layer_type='dropout',
                                            layer_depth=state.layer_depth + 1,
                                            units = 0,
                                            activation = None,
                                            proba = rate,
                                            fc_depth = state.fc_depth,
                                            fc_size=0,
                                            fc_occured = state.fc_occured,
                                            terminate=0)]

                # add fc States -- iterate through all possible fc sizes
                if state.layer_type in ['start','fc','dropout','lstm','bilstm', 'rnn', 'gru'] and state.fc_depth < self.ssp.max_fc: 
                    for fc_sizes in self._possible_fc_size(state): 
                        for activ_func in self.ssp.possible_actvf:
                            actions += [State(layer_type='fc',
                                                layer_depth=state.layer_depth + 1,
                                                fc_depth=state.fc_depth + 1,
                                                fc_size=fc_sizes,
                                                fc_occured = 1,
                                                activation = activ_func,
                                                units = 0,
                                                proba = 0,
                                                terminate=0)]

        # Add states to transition and q_value dictionary
        q_values[state.as_tuple()] = {'actions': [to_state.as_tuple() for to_state in actions],
                                      'utilities': [self.ssp.init_utility for i in range(len(actions))]}
        return q_values        

    def transition_to_action(self, start_state, to_state): 
        '''Returns a copy of state to go next'''
        action = to_state.copy()
        return action

    def state_action_transition(self, start_state, action):
        '''Returns next state given action: valid action'''
        to_state = action.copy()
        return to_state
                    
    def _possible_fc_size(self, state):
        '''
        Returns a list of possible FC sizes given the current state
        This restricts: for consecutive fc layers, the latter FC layer should not have more neurons than the FC layer before it. 
        '''
        if state.layer_type=='fc':
            return [i for i in self.ssp.possible_fc_sizes if i <= state.fc_size]
        return self.ssp.possible_fc_sizes

