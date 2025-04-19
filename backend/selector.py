import chess
import chess.engine
import json
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any, Union
import google.generativeai as genai
import os
from datetime import datetime

# Import your retrieval agent
from Retrieval import ChessRetrievalAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ChessSelectionAgent")

class ChessSelectionAgent:
    """
    Chess Selection Agent that takes retrieval results and evaluates moves
    using Stockfish and Gemini LLM to suggest the best move.
    """
    
    def __init__(
        self,
        retrieval_agent: ChessRetrievalAgent,
        stockfish_path: str,
        gemini_api_key: str,
        stockfish_depth: int = 14,
        stockfish_time_limit: float = 0.5,
        temperature: float = 0.2,
        top_k_moves: int = 3,
        prompt_template_path: Optional[str] = None
    ):
        """
        Initialize the Chess Selection Agent.
        
        Args:
            retrieval_agent: Instance of ChessRetrievalAgent
            stockfish_path: Path to Stockfish engine executable
            gemini_api_key: API key for Gemini LLM
            stockfish_depth: Depth for Stockfish analysis
            stockfish_time_limit: Time limit for Stockfish analysis in seconds
            temperature: Temperature for LLM generation
            top_k_moves: Number of top moves to consider from retrieval
            prompt_template_path: Path to prompt template file for LLM
        """
        self.retrieval_agent = retrieval_agent
        self.stockfish_path = stockfish_path
        self.stockfish_depth = stockfish_depth
        self.stockfish_time_limit = stockfish_time_limit
        self.temperature = temperature
        self.top_k_moves = top_k_moves
        
        # Initialize Stockfish engine
        logger.info(f"Initializing Stockfish engine from {stockfish_path}")
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        except Exception as e:
            logger.error(f"Failed to initialize Stockfish engine: {e}")
            raise
        
        # Initialize Gemini
        logger.info("Initializing Gemini LLM")
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
            generation_config={"temperature": temperature}
        )
        
        # Load prompt template
        self.prompt_template = self._load_prompt_template(prompt_template_path)
        
        logger.info("Chess Selection Agent initialized successfully")
    
    def _load_prompt_template(self, template_path: Optional[str]) -> str:
        """
        Load the prompt template for Gemini LLM.
        
        Args:
            template_path: Path to template file
            
        Returns:
            Prompt template string
        """
        default_template = """
        You are a chess analysis assistant helping evaluate and select the best move.
        
        Current position: {fen}
        
        Position analysis:
        {position_analysis}
        
        Top candidate moves from database:
        {candidate_moves}
        
        Engine evaluations:
        {engine_evaluations}
        
        Based on the retrieved similar positions, engine analysis, and chess principles, what is the best move in this position? 
        Explain your reasoning with a brief analysis of the top move, including any tactical or strategic themes.
        
        Provide your analysis in this format:
        - Best move: [move in SAN notation]
        - Evaluation: [simplified evaluation]
        - Reasoning: [your explanation]
        """
        
        if template_path and os.path.exists(template_path):
            try:
                with open(template_path, 'r') as f:
                    return f.read()
            except Exception as e:
                logger.warning(f"Failed to load prompt template from {template_path}: {e}")
                logger.info("Using default prompt template")
                return default_template
        else:
            logger.info("Using default prompt template")
            return default_template
    
    def evaluate_with_stockfish(self, fen: str, move: Union[str, chess.Move], time_limit: Optional[float] = None) -> Dict:
        """
        Evaluate a move with Stockfish.
        
        Args:
            fen: FEN notation of the position
            move: Move in UCI format or chess.Move object
            time_limit: Time limit for analysis in seconds (optional)
            
        Returns:
            Evaluation results
        """
        board = chess.Board(fen)
        
        # Convert string move to chess.Move if needed
        if isinstance(move, str):
            try:
                move_obj = chess.Move.from_uci(move)
            except ValueError:
                logger.error(f"Invalid move format: {move}")
                return {"error": f"Invalid move format: {move}"}
        else:
            move_obj = move
        
        # Check if move is legal
        if move_obj not in board.legal_moves:
            logger.warning(f"Illegal move: {move_obj.uci()} in position {fen}")
            return {"error": "Illegal move"}
        
        # Make the move
        board.push(move_obj)
        
        # Get engine evaluation with time limit or depth
        limit = None
        if time_limit is not None:
            limit = chess.engine.Limit(time=time_limit)
        else:
            limit = chess.engine.Limit(depth=self.stockfish_depth)
        
        try:
            info = self.engine.analyse(board, limit)
            
            # Extract score
            score = info.get("score", None)
            if score:
                # Convert score to numeric value from white's perspective
                if score.is_mate():
                    # Mate score
                    mate_score = score.mate()
                    eval_score = 999 if mate_score > 0 else -999
                else:
                    # Regular score in centipawns
                    eval_score = score.white().score() / 100.0
                    
                # Extract depth
                depth = info.get("depth", 0)
                
                return {
                    "move": move_obj.uci(),
                    "evaluation": eval_score,
                    "depth": depth,
                    "mate": score.is_mate(),
                    "mate_in": score.mate() if score.is_mate() else None
                }
            else:
                return {"error": "No score returned"}
        except Exception as e:
            logger.error(f"Error during Stockfish analysis: {e}")
            return {"error": str(e)}
    
    def evaluate_position(self, fen: str, time_limit: Optional[float] = None) -> Dict:
        """
        Evaluate the current position with Stockfish.
        
        Args:
            fen: FEN notation of the position
            time_limit: Time limit for analysis in seconds (optional)
            
        Returns:
            Position evaluation
        """
        board = chess.Board(fen)
        
        # Set limit
        limit = None
        if time_limit is not None:
            limit = chess.engine.Limit(time=time_limit)
        else:
            limit = chess.engine.Limit(depth=self.stockfish_depth)
        
        try:
            # Get both evaluation and principal variation
            info = self.engine.analyse(board, limit, multipv=1)
            
            # Extract score
            score = info.get("score", None)
            if score:
                # Convert score to numeric value
                if score.is_mate():
                    mate_score = score.mate()
                    eval_score = 999 if mate_score > 0 else -999
                    eval_text = f"Mate in {abs(mate_score)}" if mate_score > 0 else f"Mated in {abs(mate_score)}"
                else:
                    # Regular score in centipawns
                    eval_score = score.white().score() / 100.0
                    eval_text = f"{eval_score:+.2f}"
                
                # Extract principal variation if available
                pv = info.get("pv", [])
                pv_sans = []
                
                if pv:
                    temp_board = chess.Board(fen)
                    for m in pv:
                        pv_sans.append(temp_board.san(m))
                        temp_board.push(m)
                
                return {
                    "evaluation": eval_score,
                    "evaluation_text": eval_text,
                    "depth": info.get("depth", 0),
                    "pv": [m.uci() for m in pv],
                    "pv_sans": pv_sans
                }
            else:
                return {"error": "No score returned"}
        except Exception as e:
            logger.error(f"Error during position evaluation: {e}")
            return {"error": str(e)}
    
    def get_stockfish_top_moves(self, fen: str, num_moves: int = 3) -> List[Dict]:
        """
        Get top moves from Stockfish.
        
        Args:
            fen: FEN notation of the position
            num_moves: Number of top moves to return
            
        Returns:
            List of top moves with evaluations
        """
        board = chess.Board(fen)
        
        # Set limit
        limit = chess.engine.Limit(depth=self.stockfish_depth)
        
        try:
            # Get multipv analysis
            result = self.engine.analyse(board, limit, multipv=num_moves)
            
            # Process results
            moves = []
            
            # Handle multipv results
            if isinstance(result, list):
                for pvIdx, info in enumerate(result):
                    pv = info.get("pv", [])
                    if pv:
                        move = pv[0]
                        score = info.get("score", None)
                        
                        if score:
                            if score.is_mate():
                                mate_score = score.mate()
                                eval_score = 999 if mate_score > 0 else -999
                                eval_text = f"Mate in {abs(mate_score)}"
                            else:
                                eval_score = score.white().score() / 100.0
                                eval_text = f"{eval_score:+.2f}"
                            
                            moves.append({
                                "move": move.uci(),
                                "san": board.san(move),
                                "evaluation": eval_score,
                                "evaluation_text": eval_text,
                                "depth": info.get("depth", 0),
                                "rank": pvIdx + 1
                            })
            # Handle single result
            else:
                pv = result.get("pv", [])
                if pv:
                    move = pv[0]
                    score = result.get("score", None)
                    
                    if score:
                        if score.is_mate():
                            mate_score = score.mate()
                            eval_score = 999 if mate_score > 0 else -999
                            eval_text = f"Mate in {abs(mate_score)}"
                        else:
                            eval_score = score.white().score() / 100.0
                            eval_text = f"{eval_score:+.2f}"
                        
                        moves.append({
                            "move": move.uci(),
                            "san": board.san(move),
                            "evaluation": eval_score,
                            "evaluation_text": eval_text,
                            "depth": result.get("depth", 0),
                            "rank": 1
                        })
            
            return moves
            
        except Exception as e:
            logger.error(f"Error getting top moves: {e}")
            return []
    
    def rank_moves(self, fen: str, retrieved_knowledge: List[Dict]) -> List[Dict]:
        """
        Rank candidate moves based on retrieved knowledge and Stockfish evaluation.
        
        Args:
            fen: FEN notation of the position
            retrieved_knowledge: List of retrieved knowledge chunks
            
        Returns:
            Ranked list of moves with evaluations
        """
        board = chess.Board(fen)
        
        # Get all legal moves
        legal_moves = list(board.legal_moves)
        move_candidates = []
        
        # First, add moves from retrieved knowledge
        for chunk in retrieved_knowledge:
            moves = chunk.get("moves", [])
            
            # Process moves from retrieval
            for move_info in moves:
                # Handle different move info formats
                if isinstance(move_info, dict):
                    move_uci = move_info.get("move")
                    retrieval_score = move_info.get("evaluation", 0.0)
                elif hasattr(move_info, "move"):
                    move_uci = move_info.move
                    retrieval_score = getattr(move_info, "evaluation", 0.0)
                else:
                    # Try to use the move info directly as a string
                    try:
                        move_uci = str(move_info)
                        retrieval_score = 0.0
                    except:
                        continue
                
                # Check if move is legal
                try:
                    move = chess.Move.from_uci(move_uci)
                    if move in legal_moves:
                        # Get existing candidate if already present
                        existing = next((c for c in move_candidates if c["move"] == move_uci), None)
                        
                        if existing:
                            # Update existing with higher retrieval confidence
                            existing["retrieval_confidence"] = max(
                                existing["retrieval_confidence"],
                                chunk.get("score", 0.0)
                            )
                            existing["retrieval_count"] += 1
                        else:
                            # Create new candidate
                            candidate = {
                                "move": move_uci,
                                "san": board.san(move),
                                "retrieval_score": retrieval_score,
                                "retrieval_confidence": chunk.get("score", 0.0),
                                "retrieval_count": 1,
                                "source": "database"
                            }
                            move_candidates.append(candidate)
                except ValueError:
                    continue
        
        # Add Stockfish top moves if we have few candidates
        if len(move_candidates) < self.top_k_moves:
            stockfish_moves = self.get_stockfish_top_moves(fen, self.top_k_moves)
            
            for sf_move in stockfish_moves:
                move_uci = sf_move["move"]
                
                # Check if already in candidates
                existing = next((c for c in move_candidates if c["move"] == move_uci), None)
                
                if existing:
                    # Update with engine evaluation
                    existing["engine_evaluation"] = sf_move["evaluation"]
                    existing["engine_depth"] = sf_move["depth"]
                    existing["engine_rank"] = sf_move["rank"]
                else:
                    # Add new candidate from engine
                    candidate = {
                        "move": move_uci,
                        "san": sf_move["san"],
                        "engine_evaluation": sf_move["evaluation"],
                        "engine_depth": sf_move["depth"],
                        "engine_rank": sf_move["rank"],
                        "retrieval_confidence": 0.0,
                        "retrieval_count": 0,
                        "source": "engine"
                    }
                    move_candidates.append(candidate)
        
        # Evaluate all candidates with Stockfish if not already evaluated
        for candidate in move_candidates:
            if "engine_evaluation" not in candidate:
                eval_result = self.evaluate_with_stockfish(
                    fen, 
                    candidate["move"], 
                    time_limit=self.stockfish_time_limit
                )
                
                if "error" not in eval_result:
                    candidate["engine_evaluation"] = eval_result["evaluation"]
                    candidate["engine_depth"] = eval_result["depth"]
                else:
                    candidate["engine_evaluation"] = 0.0
                    candidate["engine_depth"] = 0
                    
        # Calculate final score
        # Blend engine evaluation with retrieval confidence
        for candidate in move_candidates:
            # Normalize engine evaluation to -1 to 1 range if it's extreme
            engine_eval = candidate.get("engine_evaluation", 0.0)
            if engine_eval > 10:
                engine_eval = 10.0
            elif engine_eval < -10:
                engine_eval = -10.0
                
            # Convert to 0-1 range for blending (higher is better)
            normalized_eval = (engine_eval + 10) / 20.0
            
            # Blend scores - 80% engine, 20% retrieval
            retrieval_conf = candidate.get("retrieval_confidence", 0.0)
            final_score = 0.8 * normalized_eval + 0.2 * retrieval_conf
            
            candidate["final_score"] = final_score
        
        # Rank moves by final score
        ranked_moves = sorted(move_candidates, key=lambda x: x.get("final_score", 0), reverse=True)
        
        return ranked_moves
    
    def analyze_position_with_gemini(self, fen: str, position_info: Dict, top_moves: List[Dict]) -> str:
        """
        Use Gemini LLM to analyze the position and recommend moves.
        
        Args:
            fen: FEN notation of the position
            position_info: Information about the position from retrieval
            top_moves: Top candidate moves after ranking
            
        Returns:
            LLM-generated position analysis and move recommendation
        """
        board = chess.Board(fen)
        
        # Convert position info to text
        position_analysis = f"Material: {position_info.get('material_advantage', 'Unknown')}\n"
        position_analysis += f"Game phase: {position_info.get('game_phase', 'Unknown')}\n"
        position_analysis += f"Position features: {', '.join(position_info.get('tactical_themes', ['None']))}\n"
        
        # Convert moves to text for prompt
        moves_text = ""
        for i, move in enumerate(top_moves[:3]):  # Top 3 moves
            moves_text += f"{i+1}. {move['san']} "
            if "engine_evaluation" in move:
                eval_text = f"({move['engine_evaluation']:+.2f})" if abs(move['engine_evaluation']) < 100 else "M"
                moves_text += f"- Engine evaluation: {eval_text}, "
            moves_text += f"Retrieval confidence: {move.get('retrieval_confidence', 0.0):.2f}, "
            moves_text += f"Source: {move.get('source', 'unknown')}\n"
        
        # Format engine evaluations
        engine_evals = self.get_stockfish_top_moves(fen, 3)
        engine_text = ""
        for i, move in enumerate(engine_evals):
            engine_text += f"{i+1}. {move['san']} ({move['evaluation_text']})\n"
        
        # Fill prompt template
        prompt = self.prompt_template.format(
            fen=fen,
            position_analysis=position_analysis,
            candidate_moves=moves_text,
            engine_evaluations=engine_text
        )
        
        # Generate response with Gemini
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating content with Gemini: {e}")
            return f"Error generating analysis: {str(e)}"
    
    def suggest_move(self, fen: str, human_move: Optional[str] = None) -> Dict:
        """
        Suggest a move based on retrieved knowledge, Stockfish, and Gemini LLM.
        
        Args:
            fen: FEN notation of the position
            human_move: Human's proposed move (if any)
            
        Returns:
            Dictionary with suggested move information
        """
        # 1. Retrieve relevant knowledge
        retrieved_knowledge = self.retrieval_agent.retrieve_knowledge(fen)
        
        # 2. Get position context for LLM
        position_context = self.retrieval_agent.analyze_position_context(fen)
        
        # 3. Rank candidate moves
        ranked_moves = self.rank_moves(fen, retrieved_knowledge)
        
        if not ranked_moves:
            return {"error": "No valid moves found"}
        
        # 4. Get the top move
        best_move = ranked_moves[0]
        
        # 5. If human provided a move, compare it with the best move
        human_move_analysis = None
        if human_move:
            board = chess.Board(fen)
            try:
                human_move_obj = chess.Move.from_uci(human_move)
                
                if human_move_obj in board.legal_moves:
                    # Find human move in ranked list
                    human_move_info = next((m for m in ranked_moves if m["move"] == human_move), None)
                    
                    if human_move_info is None:
                        # Evaluate the human move if not in ranked list
                        evaluation = self.evaluate_with_stockfish(fen, human_move_obj)
                        human_move_info = {
                            "move": human_move,
                            "san": board.san(human_move_obj),
                            "engine_evaluation": evaluation.get("evaluation", 0.0),
                            "engine_depth": evaluation.get("depth", 0),
                            "source": "human"
                        }
                    
                    # Compare with best move
                    best_eval = best_move.get("engine_evaluation", 0.0)
                    human_eval = human_move_info.get("engine_evaluation", 0.0)
                    
                    # Calculate difference
                    eval_diff = best_eval - human_eval
                    
                    # Adjust for player's perspective
                    if board.turn == chess.BLACK:
                        eval_diff = -eval_diff
                    
                    human_move_analysis = {
                        "move": human_move,
                        "san": human_move_info.get("san", ""),
                        "evaluation": human_eval,
                        "diff_from_best": eval_diff,
                        "rank": next((i+1 for i, m in enumerate(ranked_moves) if m["move"] == human_move), None)
                    }
                    
                    # Determine quality based on difference
                    if abs(eval_diff) < 0.1:
                        human_move_analysis["quality"] = "Excellent"
                    elif abs(eval_diff) < 0.3:
                        human_move_analysis["quality"] = "Good"
                    elif abs(eval_diff) < 0.7:
                        human_move_analysis["quality"] = "Inaccuracy"
                    elif abs(eval_diff) < 1.5:
                        human_move_analysis["quality"] = "Mistake"
                    else:
                        human_move_analysis["quality"] = "Blunder"
            except ValueError:
                human_move_analysis = {"error": "Invalid move format"}
                
        # 6. Use Gemini for analysis and explanation
        gemini_analysis = self.analyze_position_with_gemini(fen, position_context, ranked_moves)
        
        # 7. Prepare response
        result = {
            "best_move": best_move,
            "top_moves": ranked_moves[:3],
            "human_move_analysis": human_move_analysis,
            "position_context": position_context,
            "gemini_analysis": gemini_analysis,
            "timestamp": datetime.now().isoformat()
        }
        
        return result
    
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'engine') and self.engine:
            self.engine.quit()

# Example usage
def run_selection_agent(fen, human_move=None):
    """Run the selection agent with the specified file structure"""
    
    # First create the retrieval agent
    retrieval_agent = ChessRetrievalAgent(
        index_path="/Users/lakshmikamath/Desktop/tdl/knowledge-base/chess_positions.index",
        embeddings_path="/Users/lakshmikamath/Desktop/tdl/knowledge-base/raw_positions.pkl",
        metadata_path="/Users/lakshmikamath/Desktop/tdl/knowledge-base/positions_metadata.pkl",
        model_name="all-MiniLM-L6-v2",
        top_k=5
    )
    
    # Then create the selection agent
    selection_agent = ChessSelectionAgent(
        retrieval_agent=retrieval_agent,
        stockfish_path="/opt/homebrew/bin/stockfish",  # Path to Stockfish engine
        gemini_api_key="your_Api_key",  # Gemini API key
        stockfish_depth=14
    )
    
    try:
        # Get move suggestion
        result = selection_agent.suggest_move(fen, human_move)
        
        if "error" in result:
            print(f"Error: {result['error']}")
            return None
        
        # Print the result
        print(f"Position analysis:")
        print(result["gemini_analysis"])
        
        # Print top moves
        print("\nTop recommended moves:")
        for i, move in enumerate(result["top_moves"]):
            print(f"{i+1}. {move['san']} (Eval: {move.get('engine_evaluation', 0.0):+.2f})")
        
        # Print human move analysis if available
        if result["human_move_analysis"] and "error" not in result["human_move_analysis"]:
            human = result["human_move_analysis"]
            print(f"\nYour move {human['san']} is: {human['quality']}")
            print(f"Evaluation difference from best move: {human['diff_from_best']:+.2f}")
        
        return result
    finally:
        # Clean up
        selection_agent.close()

# Example with starting position
if __name__ == "__main__":
    # Starting position
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
    
    # Human wants to play e4
    human_move = "Bc4"
    
    # Get suggestion
    result = run_selection_agent(fen, human_move)