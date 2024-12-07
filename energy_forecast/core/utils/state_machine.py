from enum import Enum
from typing import Dict, Set, Optional
import logging

class DataState(Enum):
    INIT = "INIT"
    RECEIVED = "RECEIVED"
    VALIDATED = "VALIDATED"
    PROCESSED = "PROCESSED"
    ERROR = "ERROR"
    READY = "READY"

class DataEvent(Enum):
    RECEIVE = "RECEIVE"
    VALIDATE = "VALIDATE"
    PROCESS = "PROCESS"
    ERROR = "ERROR"
    COMPLETE = "COMPLETE"
    RETRY = "RETRY"

class DataStateMachine:
    """
    Finite State Automaton for data processing pipeline
    Implements a DFA (Deterministic Finite Automaton) to ensure data follows correct processing states
    """
    
    def __init__(self):
        self.current_state: DataState = DataState.INIT
        self.error_message: Optional[str] = None
        
        # Define state transitions (Q × Σ → Q)
        self.transitions: Dict[DataState, Dict[DataEvent, DataState]] = {
            DataState.INIT: {
                DataEvent.RECEIVE: DataState.RECEIVED
            },
            DataState.RECEIVED: {
                DataEvent.VALIDATE: DataState.VALIDATED,
                DataEvent.ERROR: DataState.ERROR
            },
            DataState.VALIDATED: {
                DataEvent.PROCESS: DataState.PROCESSED,
                DataEvent.ERROR: DataState.ERROR
            },
            DataState.PROCESSED: {
                DataEvent.COMPLETE: DataState.READY,
                DataEvent.ERROR: DataState.ERROR
            },
            DataState.ERROR: {
                DataEvent.RETRY: DataState.INIT
            },
            DataState.READY: {
                DataEvent.RECEIVE: DataState.RECEIVED
            }
        }
        
        # Define accepting states (F ⊆ Q)
        self.accepting_states: Set[DataState] = {DataState.READY}
        
        self.logger = logging.getLogger(__name__)
    
    def transition(self, event: DataEvent) -> bool:
        """
        Process a transition in the FSA
        Returns True if transition is valid, False otherwise
        """
        if self.current_state not in self.transitions:
            self.logger.error(f"Invalid current state: {self.current_state}")
            return False
            
        if event not in self.transitions[self.current_state]:
            self.logger.error(
                f"Invalid transition: {event} from state {self.current_state}"
            )
            return False
            
        next_state = self.transitions[self.current_state][event]
        self.logger.info(
            f"Transition: {self.current_state} -> {next_state} on event {event}"
        )
        self.current_state = next_state
        return True
    
    def is_accepted(self) -> bool:
        """Check if current state is an accepting state"""
        return self.current_state in self.accepting_states
    
    def reset(self):
        """Reset the state machine to initial state"""
        self.current_state = DataState.INIT
        self.error_message = None

class DataValidator:
    """
    Implements the validation logic using the state machine
    """
    def __init__(self):
        self.state_machine = DataStateMachine()
        self.logger = logging.getLogger(__name__)
    
    def validate_data_flow(self, data: dict) -> bool:
        """
        Validates the entire data flow using the state machine
        Returns True if data successfully reaches READY state
        """
        try:
            # Start with receiving data
            if not self.state_machine.transition(DataEvent.RECEIVE):
                return False
                
            # Validate data structure
            if not self._validate_structure(data):
                self.state_machine.transition(DataEvent.ERROR)
                return False
            
            if not self.state_machine.transition(DataEvent.VALIDATE):
                return False
                
            # Process data
            if not self._process_data(data):
                self.state_machine.transition(DataEvent.ERROR)
                return False
                
            if not self.state_machine.transition(DataEvent.PROCESS):
                return False
                
            # Complete processing
            if not self.state_machine.transition(DataEvent.COMPLETE):
                return False
            
            return self.state_machine.is_accepted()
            
        except Exception as e:
            self.logger.error(f"Error in data validation: {str(e)}")
            self.state_machine.transition(DataEvent.ERROR)
            return False
    
    def _validate_structure(self, data: dict) -> bool:
        """
        Validates the structure of input data
        Implements a Regular Expression-like pattern matching for data fields
        """
        required_fields = {
            'timestamp': lambda x: isinstance(x, str),
            'values': lambda x: isinstance(x, list),
            'metadata': lambda x: isinstance(x, dict)
        }
        
        # Check if all required fields exist and match their patterns
        for field, validator in required_fields.items():
            if field not in data or not validator(data[field]):
                self.logger.error(f"Invalid or missing field: {field}")
                return False
        
        return True
    
    def _process_data(self, data: dict) -> bool:
        """
        Processes the data after validation
        Implements a Push-down Automaton-like structure for nested data validation
        """
        try:
            stack = []
            # Validate nested structure using a stack (PDA concept)
            for item in data['values']:
                if isinstance(item, dict):
                    stack.append(item)
                    # Process nested dictionary
                    while stack:
                        current = stack.pop()
                        if not self._validate_structure(current):
                            return False
                            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in data processing: {str(e)}")
            return False
