# Fixed engine_manager.py
import chess.engine
import logging
import os
import time
from threading import Lock

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ChessEngineManager")

class ChessEngineManager:
    """
    Singleton class to manage the Stockfish chess engine.
    Ensures that only one instance of the engine is created and reused across all agents.
    """
    _instance = None
    _engine = None
    _lock = Lock()  # Add a lock for thread safety
    
    def __new__(cls, engine_path=None):
        if cls._instance is None:
            cls._instance = super(ChessEngineManager, cls).__new__(cls)
            cls._instance._engine = None
            cls._instance._engine_path = None
        
        if engine_path and not cls._instance._engine:
            cls._instance._initialize_engine(engine_path)
        
        return cls._instance
    
    def _initialize_engine(self, engine_path):
        """Initialize the Stockfish engine with retries and exponential backoff."""
        with self._lock:  # Use lock to prevent race conditions
            if self._engine is None:
                logger.info(f"Initializing Stockfish engine from {engine_path}")
                try:
                    # Check if the file exists
                    if not os.path.exists(engine_path):
                        logger.error(f"Stockfish engine not found at {engine_path}")
                        raise FileNotFoundError(f"Stockfish engine not found at {engine_path}")
                    
                    # Retry mechanism with exponential backoff
                    max_retries = 5
                    backoff = 1  # Initial backoff in seconds
                    
                    for attempt in range(1, max_retries + 1):
                        try:
                            self._engine = chess.engine.SimpleEngine.popen_uci(engine_path)
                            self._engine_path = engine_path
                            logger.info("Stockfish engine initialized successfully")
                            break
                        except Exception as e:
                            logger.warning(f"Attempt {attempt}/{max_retries} - Engine initialization error: {e}")
                            if attempt == max_retries:
                                logger.error(f"Failed to initialize Stockfish engine after {max_retries} attempts")
                                raise
                            time.sleep(backoff)
                            backoff *= 2  # Exponential backoff
                except Exception as e:
                    logger.error(f"Failed to initialize Stockfish engine: {e}")
                    raise
            else:
                logger.info("Stockfish engine already initialized")
    
    def get_engine(self):
        """Get the initialized engine instance with error handling."""
        if self._engine is None:
            if self._engine_path:
                self._initialize_engine(self._engine_path)  # Attempt to initialize the engine
            else:
                logger.error("Engine not initialized and no path provided")
                raise ValueError("Engine not initialized. Please provide a valid engine path.")
        return self._engine
    
    def analyze(self, board, limit, **kwargs):
        """Analyze a position using the engine with retries."""
        if self._engine is None:
            if self._engine_path:
                self._initialize_engine(self._engine_path)  # Attempt to initialize the engine
            else:
                logger.error("Engine not initialized and no path provided")
                raise ValueError("Engine not initialized. Please provide a valid engine path.")
        
        max_retries = 3
        backoff = 1  # Initial backoff in seconds
        
        for attempt in range(1, max_retries + 1):
            try:
                return self._engine.analyse(board, limit, **kwargs)
            except Exception as e:
                logger.error(f"Attempt {attempt}/{max_retries} - Error during position analysis: {e}")
                if attempt == max_retries:
                    logger.error("Exhausted retries for position analysis")
                    raise
                self.close()  # Close the engine and reinitialize
                self._initialize_engine(self._engine_path)
                time.sleep(backoff)
                backoff *= 2  # Exponential backoff
    
    def get_best_move(self, board, limit):
        """Get the best move for a position with retries."""
        if self._engine is None:
            if self._engine_path:
                self._initialize_engine(self._engine_path)
            else:
                logger.error("Engine not initialized and no path provided")
                raise ValueError("Engine not initialized")
        
        max_retries = 3
        backoff = 1  # Initial backoff in seconds
        
        for attempt in range(1, max_retries + 1):
            try:
                result = self._engine.play(board, limit)
                return result.move
            except Exception as e:
                logger.error(f"Attempt {attempt}/{max_retries} - Error getting best move: {e}")
                if attempt == max_retries:
                    logger.error("Exhausted retries for getting best move")
                    raise
                self.close()  # Close the engine and reinitialize
                self._initialize_engine(self._engine_path)
                time.sleep(backoff)
                backoff *= 2  # Exponential backoff
    
    def close(self):
        """Close the engine safely."""
        with self._lock:  # Use lock to prevent race conditions
            if self._engine:
                try:
                    self._engine.quit()
                except Exception as e:
                    logger.error(f"Error closing Stockfish engine: {e}")
                finally:
                    self._engine = None
                    logger.info("Stockfish engine closed")