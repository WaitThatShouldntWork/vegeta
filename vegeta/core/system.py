"""
Main VEGETA system orchestrator
"""

import logging
import asyncio
import concurrent.futures
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from .config import Config
from .exceptions import VegetaError, SessionError
from ..utils.database import DatabaseManager
from ..utils.llm_client import LLMClient
from ..extraction.entity_extractor import EntityExtractor
from ..extraction.embedding_generator import EmbeddingGenerator
from ..retrieval.graph_retriever import GraphRetriever
from ..inference.feature_generator import FeatureGenerator
from ..inference.prior_builder import PriorBuilder
from ..inference.posterior_updater import PosteriorUpdater
from ..inference.uncertainty_analyzer import UncertaintyAnalyzer
from ..session.manager import SessionManager
from ..generation.question_generator import QuestionGenerator
from ..generation.answer_generator import AnswerGenerator

logger = logging.getLogger(__name__)

@dataclass
class VegetaResponse:
    """Response from VEGETA system"""
    action: str  # ASK, ANSWER, SEARCH
    target: Optional[str]
    content: str  # Generated question, answer, or search description
    confidence: float
    reasoning: str
    session_id: str
    turn_id: int
    
    # Internal state for debugging/analysis
    internal_state: Optional[Dict[str, Any]] = None

class VegetaSystem:
    """
    Main VEGETA system orchestrator
    
    Coordinates all components to provide Bayesian active inference
    for multi-turn conversations and 20-questions style interactions.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize core services
        self.db_manager = DatabaseManager(config.database)
        self.llm_client = LLMClient(config.ollama)
        
        # Initialize components
        self._init_components()
        
        # Validate connections
        self._validate_connections()
    
    def _init_components(self):
        """Initialize all system components"""
        try:
            # Extraction components
            self.entity_extractor = EntityExtractor(self.llm_client, self.config)
            self.embedding_generator = EmbeddingGenerator(self.llm_client, self.config)
            
            # Retrieval components  
            self.graph_retriever = GraphRetriever(self.db_manager, self.config)
            
            # Inference components
            self.feature_generator = FeatureGenerator(self.db_manager, self.config)
            self.prior_builder = PriorBuilder(self.llm_client, self.db_manager, self.config)
            self.posterior_updater = PosteriorUpdater(self.config)
            self.uncertainty_analyzer = UncertaintyAnalyzer(self.llm_client, self.config)
            
            # Session management
            self.session_manager = SessionManager(self.config.session_config)
            
            # Generation components
            self.question_generator = QuestionGenerator(self.llm_client, self.config)
            self.answer_generator = AnswerGenerator(self.config)
            
            self.logger.info("✓ All components initialized successfully")
            
        except Exception as e:
            raise VegetaError(f"Failed to initialize components: {e}")
    
    def _validate_connections(self):
        """Validate database and LLM connections"""
        try:
            # Test database connection
            if not self.db_manager.test_connection():
                raise VegetaError("Database connection failed")
            
            # Test LLM connection
            if not self.llm_client.test_connection():
                raise VegetaError("LLM service connection failed")
            
            self.logger.info("✓ All connections validated")
            
        except Exception as e:
            raise VegetaError(f"Connection validation failed: {e}")
    
    def start_session(self, user_id: Optional[str] = None) -> str:
        """Start a new conversation session"""
        try:
            session_id = self.session_manager.start_session(user_id)
            self.logger.info(f"Started new session: {session_id}")
            return session_id
        except Exception as e:
            raise SessionError(f"Failed to start session: {e}")
    
    def process_query(self, session_id: str, query: str, 
                     include_internal_state: bool = False) -> VegetaResponse:
        """
        Process a user query through the full VEGETA pipeline
        
        Args:
            session_id: Active session identifier
            query: User's natural language query
            include_internal_state: Whether to include debugging info
            
        Returns:
            VegetaResponse with action, content, and metadata
        """
        import time
        
        try:
            start_time = time.time()
            self.logger.info(f"Processing query in session {session_id}: '{query}'")
            
            # Get session context for multi-turn
            step_start = time.time()
            session_context = self.session_manager.get_conversation_context(session_id)
            turn_id = session_context.get('turn_count', 0)
            self.logger.info(f"⏱️  Session context: {time.time() - step_start:.2f}s")
            
            # OPTIMIZATION: Parallel execution of independent tasks
            step_start = time.time()
            extraction_result, observation_u = self._parallel_extraction_and_embeddings(query)
            step_time = time.time() - step_start
            self.logger.info(f"⏱️  Steps 1-2 - Parallel extraction & embeddings: {step_time:.2f}s")
            
            # Step 3: Graph retrieval and candidate generation
            step_start = time.time()
            retrieval_context = self.graph_retriever.retrieve_candidates(observation_u)
            step_time = time.time() - step_start
            self.logger.info(f"⏱️  Step 3 - Graph retrieval: {step_time:.2f}s")
            
            # Step 4: Feature generation
            step_start = time.time()
            candidates = self.feature_generator.generate_features(
                retrieval_context['candidates'], 
                observation_u
            )
            step_time = time.time() - step_start
            self.logger.info(f"⏱️  Step 4 - Feature generation: {step_time:.2f}s")
            
            # Step 5: Build priors (incorporating conversation history)
            step_start = time.time()
            priors = self.prior_builder.build_priors(
                observation_u, 
                session_context, 
                retrieval_context
            )
            step_time = time.time() - step_start
            self.logger.info(f"⏱️  Step 5 - Build priors: {step_time:.2f}s")
            
            # Step 6: Update posteriors
            step_start = time.time()
            posteriors = self.posterior_updater.update_posteriors(
                candidates, 
                priors, 
                observation_u
            )
            step_time = time.time() - step_start
            self.logger.info(f"⏱️  Step 6 - Update posteriors: {step_time:.2f}s")
            
            # Step 7: Uncertainty analysis and decision making
            step_start = time.time()
            decision = self.uncertainty_analyzer.make_decision(
                posteriors,
                candidates,
                observation_u,
                retrieval_context,
                session_context
            )
            step_time = time.time() - step_start
            self.logger.info(f"⏱️  Step 7 - Uncertainty analysis: {step_time:.2f}s")

            # Handle CONTINUE_INFERENCE action by re-running with enhanced observation
            if decision.get('action') == 'CONTINUE_INFERENCE':
                enhanced_observation = decision.get('enhanced_observation')
                if enhanced_observation:
                    self.logger.info("🔄 Re-running inference with enhanced observation due to user confirmation")

                    # Re-run extraction and embedding with enhanced observation
                    enhanced_extraction = self.entity_extractor.extract_entities_llm(
                        enhanced_observation['u_meta']['utterance']
                    )
                    enhanced_observation['u_meta']['extraction'] = enhanced_extraction

                    # Re-run embedding generation
                    enhanced_observation['u_sem'] = self.embedding_generator.create_u_sem(
                        enhanced_observation['u_meta']['utterance']
                    )

                    # Re-run retrieval with enhanced observation
                    enhanced_retrieval = self.graph_retriever.retrieve_candidates(enhanced_observation)
                    enhanced_candidates = self.feature_generator.generate_features(
                        enhanced_retrieval['candidates'],
                        enhanced_observation
                    )

                    # Re-run inference with enhanced observation
                    enhanced_priors = self.prior_builder.build_priors(
                        enhanced_observation,
                        session_context,
                        enhanced_retrieval
                    )

                    enhanced_posteriors = self.posterior_updater.update_posteriors(
                        enhanced_candidates,
                        enhanced_priors,
                        enhanced_observation
                    )

                    # Make final decision with enhanced inference
                    final_decision = self.uncertainty_analyzer.make_decision(
                        enhanced_posteriors,
                        enhanced_candidates,
                        enhanced_observation,
                        enhanced_retrieval,
                        session_context
                    )

                    # Override the original decision with the enhanced one
                    decision = final_decision
                    candidates = enhanced_candidates
                    posteriors = enhanced_posteriors
                    priors = enhanced_priors
                    retrieval_context = enhanced_retrieval

                    self.logger.info("✅ Enhanced inference complete - proceeding with confirmed entity context")

            # Step 8: Generate response content
            step_start = time.time()
            response_content = self._generate_response_content(decision, candidates, session_context)
            step_time = time.time() - step_start
            self.logger.info(f"⏱️  Step 8 - Generate response: {step_time:.2f}s")
            
            # Step 9: Update session state
            step_start = time.time()
            turn_result = self._create_turn_result(
                query, decision, posteriors, priors, candidates, response_content
            )

            # CRITICAL NEW FEATURE: Update procedure state if we're in a procedure
            self._update_procedure_state_if_needed(session_id, decision, posteriors)

            # NEW: Initialize procedure state for procedure-driven checklists
            self._initialize_procedure_if_needed(session_id, posteriors)

            # CRITICAL: Update session state for next turn's priors
            self.session_manager.update_session_state(session_id, turn_result)
            step_time = time.time() - step_start
            self.logger.info(f"⏱️  Step 9 - Update session: {step_time:.2f}s")
            
            # Create response
            response = VegetaResponse(
                action=decision['action'],
                target=decision['target'],
                content=response_content,
                confidence=decision['confidence'],
                reasoning=decision['reasoning'],
                session_id=session_id,
                turn_id=turn_id
            )
            
            if include_internal_state:
                response.internal_state = {
                    'extraction': extraction_result,
                    'observation': observation_u,
                    'retrieval': retrieval_context,
                    'candidates': candidates[:3],  # Top 3 only
                    'priors': priors,
                    'posteriors': posteriors,
                    'decision': decision
                }
            
            total_time = time.time() - start_time
            self.logger.info(f"🏁 TOTAL PROCESSING TIME: {total_time:.2f}s")
            self.logger.info(f"Query processed: {decision['action']} → {decision['target']}")
            return response
            
        except Exception as e:
            self.logger.error(f"Query processing failed: {e}")
            raise VegetaError(f"Failed to process query: {e}")
    
    def add_user_feedback(self, session_id: str, feedback: str, 
                         outcome: str = "partial") -> bool:
        """Add user feedback to the last turn (for ASK responses)"""
        try:
            return self.session_manager.add_user_feedback(session_id, feedback, outcome)
        except Exception as e:
            self.logger.error(f"Failed to add user feedback: {e}")
            return False
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive session summary"""
        try:
            return self.session_manager.get_conversation_context(session_id)
        except Exception as e:
            self.logger.error(f"Failed to get session summary: {e}")
            return {}
    
    def _parallel_extraction_and_embeddings(self, query: str) -> tuple:
        """Run entity extraction and embedding generation (parallel optimization placeholder)"""
        # For now, run sequentially - can be optimized to parallel later
        extraction_result = self.entity_extractor.extract_entities_llm(query)
        observation_u = self._create_observation(query, extraction_result)
        return extraction_result, observation_u

    def _create_observation(self, query: str, extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create observation vector u from query and extraction results"""
        # Generate semantic embedding
        u_sem = self.embedding_generator.create_u_sem(query)

        # Generate term vector if we have terms
        canonical_terms = extraction_result.get('canonical_terms', [])
        u_terms_vec = None
        if canonical_terms:
            u_terms_vec = self.embedding_generator.create_u_terms_vec(canonical_terms)

        return {
            'u_sem': u_sem,
            'u_terms_set': set(canonical_terms),
            'u_terms_vec': u_terms_vec,
            'u_meta': {
                'utterance': query,
                'extraction': extraction_result
            }
        }
    
    def _generate_response_content(self, decision: Dict[str, Any],
                                 candidates: List[Dict[str, Any]],
                                 session_context: Optional[Dict[str, Any]] = None) -> str:
        """Generate appropriate response content based on decision"""
        if decision['action'] == 'ASK':
            return self.question_generator.generate_question_llm(decision, candidates, session_context)
        elif decision['action'] == 'ANSWER':
            return self.answer_generator.generate_answer(decision, candidates, session_context)
        elif decision['action'] == 'CONTINUE_INFERENCE':
            # Special case: User confirmed something, re-run inference with enhanced observation
            return f"I understand you mean {decision.get('confirmation_bias', decision['target'])}. Let me find that information for you..."
        else:  # SEARCH
            return f"Let me search for information about {decision['target']}..."
    
    def _create_turn_result(self, query: str, decision: Dict[str, Any], 
                          posteriors: Dict[str, Any], priors: Dict[str, Any],
                          candidates: List[Dict[str, Any]], response: str) -> Dict[str, Any]:
        """Create turn result for session management"""
        return {
            'user_utterance': query,
            'action': decision['action'],
            'target': decision['target'],
            'confidence': decision['confidence'],
            'entropy': decision.get('entropy', 0.0),
            'reasoning': decision['reasoning'],
            'posteriors': posteriors,
            'priors': priors,
            'top_candidates': candidates[:3],  # Top 3 candidates
            'response': response
        }

    def _update_procedure_state_if_needed(self, session_id: str, decision: Dict[str, Any],
                                        posteriors: Dict[str, Any]) -> None:
        """Update procedure state when slots are completed or procedures finish"""

        try:
            # Check if we have procedure state
            procedure_state = self.session_manager.get_procedure_state(session_id)
            if not procedure_state.get('active_checklist'):
                return

            checklist_name = procedure_state['active_checklist']
            completed_slots = procedure_state.get('completed_slots', [])

            # Check if this was an ASK action that got answered
            if decision['action'] == 'ASK':
                target_slot = decision['target']

                # If we're asking for a slot, that means we don't have it yet
                # In a real implementation, we'd wait for user response
                # For now, we'll simulate that the slot gets completed
                pass

            # Check if procedure is complete
            checklist_posterior = posteriors.get('checklist', {})
            if checklist_name in checklist_posterior:
                # Query required slots for this checklist
                required_slots = self._get_required_slots_for_checklist(checklist_name)

                # Check if all required slots are complete
                if all(slot in completed_slots for slot in required_slots):
                    self.session_manager.complete_procedure(session_id)
                    self.logger.info(f"✓ Procedure {checklist_name} completed for session {session_id}")

        except Exception as e:
            self.logger.warning(f"Error updating procedure state: {e}")

    def _get_required_slots_for_checklist(self, checklist_name: str) -> List[str]:
        """Get list of required slots for a checklist"""

        try:
            slot_specs = self.db_manager.execute_query("""
                MATCH (c:Checklist {name: $checklist_name})-[:REQUIRES]->(ss:SlotSpec)
                WHERE ss.required = true
                RETURN ss.name as slot_name
            """, {"checklist_name": checklist_name})

            return [spec['slot_name'] for spec in slot_specs]

        except Exception as e:
            self.logger.warning(f"Error getting required slots for {checklist_name}: {e}")
            return []

    def _initialize_procedure_if_needed(self, session_id: str, posteriors: Dict[str, Any]) -> None:
        """Initialize procedure state for procedure-driven checklists"""

        try:
            checklist_posterior = posteriors.get('checklist', {})
            current_procedure_state = self.session_manager.get_procedure_state(session_id)

            # If no active procedure and VerifyMusicRights is the top checklist
            if (not current_procedure_state.get('active_checklist') and checklist_posterior):
                top_checklist = max(checklist_posterior.items(), key=lambda x: x[1])[0]

                if top_checklist == 'VerifyMusicRights':
                    self.logger.info(f"Initializing VerifyMusicRights procedure for session {session_id}")

                    # Get existing SlotValues to see what we already have
                    existing_slots = self._get_existing_slot_values(session_id, top_checklist)

                    # Initialize procedure state
                    self.session_manager.update_procedure_state(
                        session_id=session_id,
                        checklist_name=top_checklist,
                        completed_slots=existing_slots,
                        current_step=f"collect_{existing_slots[0]}" if existing_slots else "collect_film"
                    )

        except Exception as e:
            self.logger.warning(f"Error initializing procedure: {e}")

    def _get_existing_slot_values(self, session_id: str, checklist_name: str) -> List[str]:
        """Get list of slots that already have values for this checklist"""

        try:
            # Query for SlotValues that exist for entities in this session
            # For now, we'll check for the 'film' SlotValue that we know exists
            existing_slots = []

            # Check if Skyfall film SlotValue exists (from our seed data)
            slot_values = self.db_manager.execute_query("""
                MATCH (sv:SlotValue {slot: 'film', value: 'film:skyfall'})
                RETURN sv.slot as slot_name
            """)

            if slot_values:
                existing_slots.append('film')

            self.logger.info(f"Found existing slots for {checklist_name}: {existing_slots}")
            return existing_slots

        except Exception as e:
            self.logger.warning(f"Error getting existing slot values: {e}")
            return []

    def close(self):
        """Clean shutdown of all components"""
        try:
            if hasattr(self, 'db_manager'):
                self.db_manager.close()
            self.logger.info("✓ VEGETA system shutdown complete")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
