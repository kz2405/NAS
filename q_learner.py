import numpy as np
import pandas as pd
import os
import netparser 
import state_enumerator as se
from  state_string_utils import StateStringUtils

class QValues:
    ''' Stores Q_values as a dict {start_states, {actions, Q_values}} with helper functions.'''
    def __init__(self):
        self.q = {}

    def load_q_values(self, q_csv_path):
        '''Loads Q_values from a csv file'''
        self.q = {}
        q_csv = pd.read_csv(q_csv_path)
        for row in zip(*[q_csv[col].values.tolist() for col in ['start_layer_type',
                                              'start_layer_depth',
                                              'start_units',
                                              'start_activation',
                                              'start_proba',
                                              'start_fc_depth',
                                              'start_fc_size',
                                              'start_fc_occured',
                                              'start_terminate',
                                              'end_layer_type',
                                              'end_layer_depth',
                                              'end_units',
                                              'end_activation',
                                              'end_proba',
                                              'end_fc_depth',
                                              'end_fc_size',
                                              'end_fc_occured',
                                              'end_terminate',
                                              'utility']]):
            start_state = se.State(layer_type = row[0],
                                   layer_depth = row[1],
                                   units = row[2],
                                   activation = row[3] if not pd.isnull(row[3])  else None,
                                   proba = row[4],
                                   fc_depth = row[5],
                                   fc_size = row[6],
                                   fc_occured = row[7],
                                   terminate = row[8]).as_tuple()
            end_state = se.State(layer_type = row[9],
                                 layer_depth = row[10],
                                 units = row[11],
                                 activation = row[12] if not pd.isnull(row[12])  else None,
                                 proba = row[13],
                                 fc_depth = row[14],
                                 fc_size = row[15],
                                 fc_occured = row[16],
                                 terminate = row[17]).as_tuple()
            utility = row[18]

            if start_state not in self.q:
                self.q[start_state] = {'actions': [end_state], 'utilities': [utility]}
            else:
                self.q[start_state]['actions'].append(end_state)
                self.q[start_state]['utilities'].append(utility)


    def save_to_csv(self, q_csv_path):
        '''writes Q_values to a csv file'''
        start_layer_type = []
        start_layer_depth = []
        start_units = []
        start_activation = []
        start_proba = []
        start_fc_depth = []
        start_fc_size = []
        start_fc_occured = []
        start_terminate = []
        end_layer_type = []
        end_layer_depth = []
        end_units = []
        end_activation = []
        end_proba = []
        end_fc_depth = []
        end_fc_size = []
        end_fc_occured = []
        end_terminate = []
        utility = []
        for start_state_list in self.q.keys():
            start_state = se.State(state_list=start_state_list)
            for to_state_ix in range(len(self.q[start_state_list]['actions'])):
                to_state = se.State(state_list=self.q[start_state_list]['actions'][to_state_ix])
                utility.append(self.q[start_state_list]['utilities'][to_state_ix])
                start_layer_type.append(start_state.layer_type)
                start_layer_depth.append(start_state.layer_depth)
                start_units.append(start_state.units)
                start_activation.append(start_state.activation)
                start_proba.append(start_state.proba)
                start_fc_depth.append(start_state.fc_depth)
                start_fc_size.append(start_state.fc_size)
                start_fc_occured.append(start_state.fc_occured)
                start_terminate.append(start_state.terminate)
                end_layer_type.append(to_state.layer_type)
                end_layer_depth.append(to_state.layer_depth)
                end_units.append(to_state.units)
                end_activation.append(to_state.activation)
                end_proba.append(to_state.proba)
                end_fc_depth.append(to_state.fc_depth)
                end_fc_size.append(to_state.fc_size)
                end_fc_occured.append(to_state.fc_occured)
                end_terminate.append(to_state.terminate)

        q_csv = pd.DataFrame({'start_layer_type' : start_layer_type,
                              'start_layer_depth' : start_layer_depth,
                              'start_units' : start_units,
                              'start_activation' : start_activation,
                              'start_proba' : start_proba,
                              'start_fc_depth' : start_fc_depth,
                              'start_fc_size' : start_fc_size,
                              'start_fc_occured' : start_fc_occured,
                              'start_terminate' : start_terminate,
                              'end_layer_type' : end_layer_type,
                              'end_layer_depth' : end_layer_depth,
                              'end_units' : end_units,
                              'end_activation' : end_activation,
                              'end_proba' : end_proba,
                              'end_fc_depth' : end_fc_depth,
                              'end_fc_size' : end_fc_size,
                              'end_fc_occured' : end_fc_occured,
                              'end_terminate' : end_terminate,
                              'utility' : utility})
        q_csv.to_csv(q_csv_path, index=False)


class QLearner:
    ''' Class that handles Q_learning through Q_value updates and architecture generation'''
    
    def __init__(self,
                 state_space_parameters, 
                 epsilon,
                 state=None,
                 qstore=None,
                 replay_dictionary=pd.DataFrame(columns=['net',
                                                         'accuracy_best_val',
                                                         'epsilon'])):
        self.state_list = []

        self.state_space_parameters = state_space_parameters

        # Class that will expand states for us
        self.enum = se.StateEnumerator(state_space_parameters)
        self.stringutils = StateStringUtils(state_space_parameters)

        # Starting State
        self.state = se.State('start', 0, 0, None, 0, 0, 0, 0, 0) if not state else state

        # Q Values
        self.qstore = QValues() if not qstore else qstore
        
        # Replay database which stores the RMSE of all trained models
        self.replay_dictionary = replay_dictionary

        self.epsilon=epsilon 

    def update_replay_database(self, new_replay_dic):
        '''updates the replay database'''
        self.replay_dictionary = new_replay_dic

    def generate_net(self):
        ''' Have Q-Learning agent generate a network using current Q_values and convert network to string format'''
        self._reset_for_new_walk()
        state_list = self._run_agent()
        net_string = self.stringutils.state_list_to_string(state_list)

        # Check if we have already trained this model
        if net_string in self.replay_dictionary['net'].values:
            acc_best_val = self.replay_dictionary[self.replay_dictionary['net']==net_string]['accuracy_best_val'].values[0]
            
        else:

            acc_best_val = -1.0

        return (net_string, acc_best_val)

    def save_q(self, q_path):
        '''calls the save to csv method of QValues class'''
        self.qstore.save_to_csv(os.path.join(q_path,'q_values.csv'))

    def _reset_for_new_walk(self):
        '''Reset the state for a new random walk'''
        # Architecture String
        self.state_list = []

        # Starting State
        self.state = se.State('start', 0, 0, None, 0, 0, 0, 0, 0) #to check

    def _run_agent(self):
        ''' Have Q-Learning agent use current epsilon and QValues to generate a network'''
        while self.state.terminate == 0:
            self._transition_q_learning()

        return self.state_list

    def _transition_q_learning(self):
        '''Updates next state according to an epsilon-greedy strategy'''
        if self.state.as_tuple() not in self.qstore.q:
            self.enum.enumerate_state(self.state, self.qstore.q)

        action_values = self.qstore.q[self.state.as_tuple()]
        
        # epsilon greedy choice
        if np.random.random() < self.epsilon:
            possiblelayertypes = list({layer[0] for layer in action_values['actions']})
            chosenlayertype = possiblelayertypes[np.random.randint(len(possiblelayertypes))]
            possiblelayers = [layer for layer in action_values['actions'] if layer[0] == chosenlayertype]
            action = se.State(state_list=possiblelayers[np.random.randint(len(possiblelayers))])
        else:
            min_q_value = min(action_values['utilities'])
            min_q_indexes = [i for i in range(len(action_values['actions'])) if action_values['utilities'][i]==min_q_value]
            min_actions = [action_values['actions'][i] for i in min_q_indexes]
            action = se.State(state_list=min_actions[np.random.randint(len(min_actions))])

        self.state = self.enum.state_action_transition(self.state, action)

        self._post_transition_updates()

    def _post_transition_updates(self):
        '''updates class state list'''

        self.state_list.append(self.state.copy())

    def sample_replay_for_update(self):
        '''Sample from replay database to update QValues'''
        for i in range(self.state_space_parameters.replay_number):
            net = np.random.choice(self.replay_dictionary['net'])
            accuracy_best_val = self.replay_dictionary[self.replay_dictionary['net'] == net]['accuracy_best_val'].values[0]
            state_list = self.stringutils.convert_model_string_to_states(netparser.parse('net', net))
            
            self.update_q_value_sequence(state_list, self.accuracy_to_reward(accuracy_best_val))

    def accuracy_to_reward(self, acc):
        '''How to define reward from accuracy (in our case, accuracy = RMSE)'''
        return acc

    def update_q_value_sequence(self, states, termination_reward):
        '''Update a sequence of QValues corresponding to the trained architecture'''
        self._update_q_value(states[-2], states[-1], termination_reward)
        for i in reversed(range(len(states) - 2)):
            self._update_q_value(states[i], states[i+1], 0)

    def _update_q_value(self, start_state, to_state, reward):
        ''' Update a single Q-Value for start_state given the state we transitioned to and the reward. '''
        if start_state.as_tuple() not in self.qstore.q:
            self.enum.enumerate_state(start_state, self.qstore.q)
        if to_state.as_tuple() not in self.qstore.q:
            self.enum.enumerate_state(to_state, self.qstore.q)

        actions = self.qstore.q[start_state.as_tuple()]['actions']
        values = self.qstore.q[start_state.as_tuple()]['utilities']

        min_over_next_states = min(self.qstore.q[to_state.as_tuple()]['utilities']) if to_state.terminate != 1 else 0

        action_between_states = to_state.as_tuple()

        # Q_Learning update rule
        values[actions.index(action_between_states)] = values[actions.index(action_between_states)] + \
                                                self.state_space_parameters.learning_rate * (reward + self.state_space_parameters.discount_factor * min_over_next_states - values[actions.index(action_between_states)])

        self.qstore.q[start_state.as_tuple()] = {'actions': actions, 'utilities': values}

    




