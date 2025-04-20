import chess
import chess.engine
import json
import logging
from typing import List, Dict, Optional, Union
import google.generativeai as genai
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ChessDebateAgent")

class ChessDebateAgent:
    """
    Chess Debate Agent that presents multiple perspectives on chess positions and moves.
    It can generate discussions between different chess styles or approaches.
    """
    
    def __init__(
        self,
        stockfish_path: str,
        gemini_api_key: str,
        stockfish_depth: int = 14,
        temperature: float = 0.7,  # Higher temperature for more creative debates
        debate_styles: Optional[List[str]] = None
    ):
        """
        Initialize the Chess Debate Agent.
        
        Args:
            stockfish_path: Path to Stockfish engine executable
            gemini_api_key: API key for Gemini LLM
            stockfish_depth: Depth for Stockfish analysis
            temperature: Temperature for LLM generation
            debate_styles: List of chess styles/approaches for the debate
        """
        self.stockfish_path = stockfish_path
        self.stockfish_depth = stockfish_depth
        self.temperature = temperature
        
        # Default debate styles if none provided
        self.debate_styles = debate_styles or [
            "Classical Positional",
            "Tactical Attacker", 
            "Modern Dynamic",
            "Endgame Specialist"
        ]
        
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
        
        logger.info("Chess Debate Agent initialized successfully")
    
    def get_stockfish_evaluation(self, fen: str) -> Dict:
        """
        Get Stockfish evaluation of the position.
        
        Args:
            fen: FEN notation of the position
            
        Returns:
            Position evaluation details
        """
        board = chess.Board(fen)
        
        # Set analysis limit
        limit = chess.engine.Limit(depth=self.stockfish_depth)
        
        try:
            # Get the evaluation and principal variation
            info = self.engine.analyse(board, limit, multipv=3)
            
            # Process results for multiple lines
            if isinstance(info, list):
                lines = []
                for i, line_info in enumerate(info):
                    pv = line_info.get("pv", [])
                    score = line_info.get("score", None)
                    
                    if score and pv:
                        # Convert score to text
                        if score.is_mate():
                            mate_score = score.mate()
                            eval_text = f"Mate in {abs(mate_score)}" if mate_score > 0 else f"Mated in {abs(mate_score)}"
                        else:
                            eval_score = score.white().score() / 100.0
                            eval_text = f"{eval_score:+.2f}"
                        
                        # Convert PV to SAN notation
                        pv_sans = []
                        temp_board = chess.Board(fen)
                        for m in pv:
                            pv_sans.append(temp_board.san(m))
                            temp_board.push(m)
                        
                        lines.append({
                            "rank": i + 1,
                            "evaluation": eval_text,
                            "moves": " ".join(pv_sans[:5])  # First 5 moves in the line
                        })
                
                return {
                    "top_lines": lines,
                    "depth": info[0].get("depth", 0) if info else 0
                }
            else:
                # Single line analysis
                pv = info.get("pv", [])
                score = info.get("score", None)
                
                if score and pv:
                    # Convert score to text
                    if score.is_mate():
                        mate_score = score.mate()
                        eval_text = f"Mate in {abs(mate_score)}" if mate_score > 0 else f"Mated in {abs(mate_score)}"
                    else:
                        eval_score = score.white().score() / 100.0
                        eval_text = f"{eval_score:+.2f}"
                    
                    # Convert PV to SAN notation
                    pv_sans = []
                    temp_board = chess.Board(fen)
                    for m in pv:
                        pv_sans.append(temp_board.san(m))
                        temp_board.push(m)
                    
                    return {
                        "top_lines": [{
                            "rank": 1,
                            "evaluation": eval_text,
                            "moves": " ".join(pv_sans[:5])
                        }],
                        "depth": info.get("depth", 0)
                    }
                else:
                    return {"error": "No analysis available"}
        except Exception as e:
            logger.error(f"Error during Stockfish analysis: {e}")
            return {"error": str(e)}
    
    def _create_debate_prompt(self, fen: str, stockfish_eval: Dict, num_perspectives: int = 2) -> str:
        """
        Create a prompt for the debate.
        
        Args:
            fen: FEN notation of the position
            stockfish_eval: Stockfish evaluation results
            num_perspectives: Number of perspectives to include in the debate
            
        Returns:
            Prompt for Gemini LLM
        """
        board = chess.Board(fen)
        
        # Determine material count and game phase
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9
        }
        
        white_material = sum(len(list(board.pieces(piece, chess.WHITE))) * value 
                          for piece, value in piece_values.items())
        black_material = sum(len(list(board.pieces(piece, chess.BLACK))) * value 
                          for piece, value in piece_values.items())
        
        total_material = white_material + black_material
        
        # Determine game phase
        if total_material >= 62:  # Full material is 78
            game_phase = "Opening"
        elif total_material >= 30:
            game_phase = "Middlegame"
        else:
            game_phase = "Endgame"
            
        # Get move number and side to move
        move_number = board.fullmove_number
        side_to_move = "White" if board.turn else "Black"
        
        # Select styles for the debate
        if num_perspectives > len(self.debate_styles):
            num_perspectives = len(self.debate_styles)
            
        debate_styles = self.debate_styles[:num_perspectives]
        
        # Create the prompt
        prompt = f"""You are hosting a debate between {num_perspectives} chess experts with different playing styles and approaches. They're analyzing the following position:

FEN: {fen}
Game phase: {game_phase}
Move number: {move_number}
Side to move: {side_to_move}
Material balance: White has {white_material} points, Black has {black_material} points.

The Stockfish engine evaluates this position as follows:
"""

        # Add Stockfish evaluations
        for line in stockfish_eval.get("top_lines", []):
            prompt += f"Line {line['rank']}: {line['evaluation']} - {line['moves']}\n"
            
        prompt += f"\nNow I want you to create a debate between these {num_perspectives} chess experts with different styles:\n"
        
        for style in debate_styles:
            prompt += f"- {style}\n"
            
        prompt += f"""
Each expert should:
1. Briefly assess the position from their stylistic perspective
2. Suggest what they believe is the best move or plan
3. Explain why their approach is superior in this particular position
4. Respond to or critique other experts' suggestions

Format the debate as a conversation with each expert's name followed by their analysis. 
Make sure their recommendations are consistent with their style while still being sound chess advice.
Ensure each expert has a distinct perspective and reasoning that reflects their chess philosophy.

Keep the debate focused, insightful, and with specific move suggestions that would actually make sense in this position.
"""

        return prompt
    
    def generate_debate(self, fen: str, num_perspectives: int = 2) -> Dict:
        """
        Generate a debate between different chess perspectives.
        
        Args:
            fen: FEN notation of the position
            num_perspectives: Number of perspectives to include
            
        Returns:
            Dictionary with debate content and metadata
        """
        # Validate FEN
        try:
            chess.Board(fen)
        except ValueError:
            return {"error": "Invalid FEN position"}
            
        # Get Stockfish evaluation
        stockfish_eval = self.get_stockfish_evaluation(fen)
        
        if "error" in stockfish_eval:
            return {"error": f"Stockfish evaluation failed: {stockfish_eval['error']}"}
            
        # Create debate prompt
        prompt = self._create_debate_prompt(fen, stockfish_eval, num_perspectives)
        
        # Generate debate with Gemini
        try:
            response = self.model.generate_content(prompt)
            
            return {
                "debate": response.text,
                "stockfish_evaluation": stockfish_eval,
                "perspectives": self.debate_styles[:num_perspectives],
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error generating debate: {e}")
            return {"error": f"Failed to generate debate: {str(e)}"}
    
    def analyze_move_debate(self, fen: str, move: str, num_perspectives: int = 2) -> Dict:
        """
        Generate a debate specifically about a proposed move.
        
        Args:
            fen: FEN notation of the position
            move: The move to debate (UCI format)
            num_perspectives: Number of perspectives to include
            
        Returns:
            Dictionary with debate content focused on the move
        """
        # Validate FEN and move
        try:
            board = chess.Board(fen)
            move_obj = chess.Move.from_uci(move)
            
            if move_obj not in board.legal_moves:
                return {"error": "Illegal move"}
                
            move_san = board.san(move_obj)
        except ValueError:
            return {"error": "Invalid FEN position or move format"}
        except Exception as e:
            return {"error": f"Error: {str(e)}"}
            
        # Get Stockfish evaluation of the position
        stockfish_eval = self.get_stockfish_evaluation(fen)
        
        if "error" in stockfish_eval:
            return {"error": f"Stockfish evaluation failed: {stockfish_eval['error']}"}
        
        # Also get evaluation after the move
        board.push(move_obj)
        post_move_eval = self.get_stockfish_evaluation(board.fen())
        
        # Create debate prompt specifically about the move
        debate_styles = self.debate_styles[:num_perspectives]
        
        prompt = f"""You are hosting a debate between {num_perspectives} chess experts with different playing styles about a specific move that was just played.

The move {move_san} was played in this position:
FEN: {fen}

The Stockfish engine evaluated the position before the move as:
"""

        # Add pre-move evaluations
        for line in stockfish_eval.get("top_lines", []):
            prompt += f"Line {line['rank']}: {line['evaluation']} - {line['moves']}\n"
            
        prompt += f"\nAfter the move {move_san}, the evaluation changed to:\n"
        
        # Add post-move evaluations
        for line in post_move_eval.get("top_lines", []):
            prompt += f"Line {line['rank']}: {line['evaluation']} - {line['moves']}\n"
            
        prompt += f"\nNow I want you to create a debate between these {num_perspectives} chess experts with different styles:\n"
        
        for style in debate_styles:
            prompt += f"- {style}\n"
            
        prompt += f"""
Each expert should:
1. Evaluate the move {move_san} from their stylistic perspective
2. Discuss whether they would have played this move and why/why not
3. Suggest what they might have played instead, if different
4. Explain the strategic or tactical implications of the move
5. Respond to other experts' points about the move

Format the debate as a conversation with each expert's name followed by their analysis.
Ensure each expert has a distinct perspective that reflects their chess philosophy while providing sound chess analysis.
Keep the debate focused specifically on the move {move_san} and its implications.
"""

        # Generate debate with Gemini
        try:
            response = self.model.generate_content(prompt)
            
            return {
                "move": move_san,
                "debate": response.text,
                "stockfish_evaluation_before": stockfish_eval,
                "stockfish_evaluation_after": post_move_eval,
                "perspectives": debate_styles,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error generating move debate: {e}")
            return {"error": f"Failed to generate debate: {str(e)}"}
    
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'engine') and self.engine:
            self.engine.quit()