# analyzer.py
import chess
import chess.engine
import logging
from typing import Dict, Optional, List, Any, Union
import google.generativeai as genai
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("HumanMoveAnalysisAgent")

class HumanMoveAnalysisAgent:
    def __init__(
        self,
        stockfish_path: str,
        gemini_api_key: str,
        stockfish_depth: int = 14,
        temperature: float = 0.3,
        move_quality_thresholds: Optional[Dict[str, float]] = None,
        gemini_model: str = "gemini-1.5-pro"
    ):
        self.stockfish_path = stockfish_path
        self.stockfish_depth = stockfish_depth
        self.temperature = temperature
        self.gemini_model = gemini_model
        self.move_quality_thresholds = move_quality_thresholds or {
            "Excellent": 0.1,
            "Good": 0.3,
            "Reasonable": 0.7,
            "Suboptimal": 1.0,
            "Bad": 1.5,
            "Blunder": float('inf')
        }

        # Initialize Gemini
        logger.info(f"Initializing Gemini with model: {gemini_model}")
        try:
            genai.configure(api_key=gemini_api_key)
            self.gemini = genai.GenerativeModel(
                model_name=gemini_model,
                generation_config={"temperature": temperature}
            )
            logger.info("Gemini API initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini API: {e}")
            raise

        # Initialize Stockfish
        logger.info(f"Initializing Stockfish engine from {stockfish_path}")
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
            logger.info("Stockfish initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Stockfish engine: {e}")
            self.engine = None
            raise

    def analyze_human_move(self, fen: str, move: str, move_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Analyze a human move using Stockfish and Gemini.
        
        Args:
            fen: Position FEN string before the move was made
            move: The UCI format move made by the human
            move_history: Optional list of previous moves in the game
            
        Returns:
            Dictionary containing the analysis results
        """
        board = chess.Board(fen)
        
        try:
            # Validate the move
            move_obj = chess.Move.from_uci(move)
            if move_obj not in board.legal_moves:
                return {"error": "Illegal move"}
            
            # Get SAN notation for better readability
            move_san = board.san(move_obj)
            
            # Create a new board to represent position after the move
            position_after_move = board.copy()
            position_after_move.push(move_obj)
            
            # Analyze with Stockfish
            stockfish_analysis = self._evaluate_with_stockfish(fen, move)
            if "error" in stockfish_analysis:
                return stockfish_analysis
            
            # Determine move quality based on Stockfish evaluation
            stockfish_quality = self._determine_move_quality(stockfish_analysis["evaluation_diff"])
            
            # Format move history for analysis context
            formatted_history = self._format_move_history(move_history) if move_history else []
            
            # Analyze with Gemini
            gemini_analysis = self._analyze_with_gemini(
                board, 
                move_san, 
                stockfish_analysis, 
                stockfish_quality,
                position_after_move,
                formatted_history
            )
            
            # Build the complete analysis
            return {
                "move": move,
                "move_san": move_san,
                "stockfish_analysis": stockfish_analysis,
                "stockfish_quality": stockfish_quality,
                "gemini_analysis": gemini_analysis,
                "classification": gemini_analysis.get("classification", stockfish_quality),
                "explanation": gemini_analysis.get("explanation", ""),
                "impact_assessment": gemini_analysis.get("impact_assessment", ""),
                "suggested_improvements": gemini_analysis.get("suggested_improvements", "")
            }
            
        except ValueError as e:
            return {"error": f"Invalid move format: {str(e)}"}
        except Exception as e:
            logger.error(f"Error analyzing move: {e}")
            return {"error": f"Analysis error: {str(e)}"}

    def _evaluate_with_stockfish(self, fen: str, move: Union[str, chess.Move]) -> Dict[str, Any]:
        """
        Evaluate a position and move using the Stockfish engine
        
        Returns a dictionary with evaluation details or an error
        """
        board = chess.Board(fen)
        try:
            # Convert move to move object if it's a string
            move_obj = chess.Move.from_uci(move) if isinstance(move, str) else move
            
            # Set analysis parameters
            limit = chess.engine.Limit(depth=self.stockfish_depth)
            
            # Get best move in current position
            best_move_result = self.engine.analyse(board, limit, multipv=3)
            if isinstance(best_move_result, list):
                best_move_result = best_move_result[0]  # Take first (best) variation
                
            best_move = best_move_result["pv"][0] if "pv" in best_move_result and best_move_result["pv"] else None
            best_move_score = best_move_result.get("score", None)
            
            if not best_move or not best_move_score:
                return {"error": "Failed to get best move from Stockfish"}
            
            # Get best move SAN notation BEFORE making any moves on the board
            best_move_san = board.san(best_move)
            
            # Get top alternative moves
            top_moves = []
            multi_pv_result = self.engine.analyse(board, limit, multipv=3)
            if isinstance(multi_pv_result, list):
                for idx, pvs in enumerate(multi_pv_result):
                    if "pv" in pvs and pvs["pv"]:
                        top_move = pvs["pv"][0]
                        top_moves.append({
                            "move": top_move.uci(),
                            "san": board.san(top_move),  # Get SAN before modifying board
                            "eval": self._convert_score_to_value(pvs.get("score"))
                        })
            
            # Make the human move and evaluate the resulting position
            board.push(move_obj)
            after_move_result = self.engine.analyse(board, limit)
            after_move_score = after_move_result.get("score", None)
            
            if not after_move_score:
                return {"error": "Failed to evaluate position after move"}
                
            # Format the analysis result - pass best_move_san instead of calculating it later
            return self._format_stockfish_analysis(best_move, best_move_san, best_move_score, after_move_score, board, top_moves)
            
        except Exception as e:
            logger.error(f"Error during Stockfish analysis: {e}")
            return {"error": f"Stockfish analysis error: {str(e)}"}

    def _format_stockfish_analysis(self, best_move, best_move_san, best_move_score, after_move_score, board, top_moves=None):
        """Format the Stockfish analysis results into a structured dictionary"""
        perspective_factor = 1 if board.turn == chess.BLACK else -1
        best_move_value = self._convert_score_to_value(best_move_score)
        after_move_value = self._convert_score_to_value(after_move_score)
        eval_diff = (best_move_value - after_move_value) * perspective_factor

        result = {
            "best_move": best_move.uci(),
            "best_move_san": best_move_san,  # Use the provided SAN instead of calculating it again
            "best_move_eval": best_move_value,
            "played_move_eval": after_move_value,
            "evaluation_diff": eval_diff,
            "position_advantage": "white" if after_move_value > 0 else "black" if after_move_value < 0 else "equal",
            "advantage_magnitude": abs(after_move_value)
        }
        
        if top_moves:
            result["top_moves"] = top_moves
            
        return result

    def _convert_score_to_value(self, score):
        try:
            perspective_score = score.white()  # Always use White's POV for consistency
            if perspective_score.is_mate():
                mate_value = perspective_score.mate()
                return (10000 - abs(mate_value)) * (1 if mate_value > 0 else -1)
            return perspective_score.score() / 100.0
        except Exception as e:
            logger.error(f"Error converting score to value: {e}, score type: {type(score)}")
            try:
                return 2.0 * (score.wdl().winning_chance() - 0.5) * 10.0
            except:
                return 0.0

    def _determine_move_quality(self, eval_diff: float) -> str:
        """Determine the quality of a move based on the evaluation difference"""
        # Special handling for mate scores
        if abs(eval_diff) > 9000:  # This is likely related to mate score differences
            # Lost a mate or extended mate sequence significantly
            return "Blunder"
        elif abs(eval_diff) > 5000:  # Still in mate territory but significant difference
            return "Bad"
            
        # Normal evaluation thresholds for non-mate positions
        for quality, threshold in sorted(self.move_quality_thresholds.items(), key=lambda x: x[1]):
            if eval_diff <= threshold:
                return quality
        return "Blunder"  # Default if none matched

    def _analyze_with_gemini(self, board, move_san, stockfish_analysis, stockfish_quality, position_after_move, move_history):
        """
        Analyze a chess move using Gemini AI
        
        Returns a structured analysis of the move
        """
        try:
            # Create a prompt for Gemini
            prompt = self._create_gemini_prompt(
                board, 
                move_san, 
                stockfish_analysis, 
                stockfish_quality, 
                position_after_move,
                move_history
            )
            
            # Get the response from Gemini
            response = self.gemini.generate_content(prompt)
            
            # Parse the response
            analysis_text = response.text
            
            # Extract structured information using another Gemini call
            structured_analysis = self._extract_structured_analysis(analysis_text, stockfish_quality)
            
            return structured_analysis
            
        except Exception as e:
            logger.error(f"Error during Gemini analysis: {e}")
            return {
                "classification": stockfish_quality,
                "explanation": f"Gemini analysis failed: {str(e)}. Stockfish evaluated this move as {stockfish_quality}.",
                "impact_assessment": "Unable to assess impact due to analysis error."
            }

    def _create_gemini_prompt(self, board, move_san, stockfish_analysis, stockfish_quality, position_after_move, move_history):
        """Create a detailed prompt for Gemini to analyze a chess move"""
        
        # Chess position details
        fen = board.fen()
        position_after_fen = position_after_move.fen()
        turn = "White" if board.turn == chess.WHITE else "Black"
        move_number = board.fullmove_number
        
        # Game state information
        is_check = board.is_check()
        is_checkmate = board.is_checkmate()
        is_stalemate = board.is_stalemate()
        is_insufficient_material = board.is_insufficient_material()
        material_difference = self._calculate_material_difference(board)
        
        # Format move history
        history_text = "\n".join([f"{i+1}. {move['san']}" for i, move in enumerate(move_history)])
        if not history_text:
            history_text = "No previous moves available"
        
        prompt = f"""
You are a master chess coach analyzing a human player's move. Provide insightful analysis about the move and its implications.

CHESS POSITION INFORMATION:
- Current position (FEN): {fen}
- Position after move (FEN): {position_after_fen}
- Turn: {turn}
- Move number: {move_number}
- Move played: {move_san}
- Game state: {"Check" if is_check else "Not in check"}, {"Checkmate" if is_checkmate else "Not checkmate"}, {"Stalemate" if is_stalemate else "Not stalemate"}
- Material difference: {material_difference}

STOCKFISH ANALYSIS:
- Best move according to Stockfish: {stockfish_analysis['best_move_san']}
- Evaluation before move: {stockfish_analysis['best_move_eval']}
- Evaluation after move: {stockfish_analysis['played_move_eval']}
- Evaluation difference: {stockfish_analysis['evaluation_diff']}
- Stockfish quality assessment: {stockfish_quality}

MOVE HISTORY:
{history_text}

Please analyze this move comprehensively covering:
1. A classification of the move (Excellent, Good, Reasonable, Suboptimal, Bad, or Blunder)
2. A detailed explanation of the move's strengths and weaknesses
3. The strategic and tactical implications
4. How this move impacts the overall game position
5. What concepts or principles the move demonstrates or violates
6. Better alternatives if applicable
7. Learning opportunities from this move

Format your analysis as an expert coach would explain to a player looking to improve.
"""
        return prompt

    def _extract_structured_analysis(self, analysis_text, stockfish_quality):
        """Extract structured information from Gemini's analysis text"""
        
        prompt = f"""
Given the following chess move analysis, extract structured information into a JSON format:

{analysis_text}

Return a JSON object with these keys:
- classification: The move quality (Excellent, Good, Reasonable, Suboptimal, Bad, or Blunder)
- explanation: A concise explanation of the move's strengths and weaknesses
- impact_assessment: How this move impacts the position
- suggested_improvements: Better alternatives or learning opportunities

If any section is unclear from the analysis, use this default classification: {stockfish_quality}
Format as valid JSON only.
"""
        
        try:
            response = self.gemini.generate_content(prompt)
            response_text = response.text
            
            # Extract JSON from the response
            if '```json' in response_text:
                json_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                json_text = response_text.split('```')[1].split('```')[0].strip()
            else:
                json_text = response_text.strip()
                
            structured_data = json.loads(json_text)
            return structured_data
            
        except Exception as e:
            logger.warning(f"Failed to extract structured analysis: {e}")
            return {
                "classification": stockfish_quality,
                "explanation": analysis_text[:500] + "..." if len(analysis_text) > 500 else analysis_text,
                "impact_assessment": "Unable to structure the analysis.",
                "suggested_improvements": "See full analysis text."
            }

    def _calculate_material_difference(self, board):
        """Calculate the material difference in the position"""
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9
        }
        
        white_material = 0
        black_material = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values.get(piece.piece_type, 0)
                if piece.color == chess.WHITE:
                    white_material += value
                else:
                    black_material += value
                    
        difference = white_material - black_material
        if difference > 0:
            return f"White is ahead by {difference} points"
        elif difference < 0:
            return f"Black is ahead by {abs(difference)} points"
        else:
            return "Material is equal"

    def _format_move_history(self, move_history):
        """Format the move history for analysis"""
        formatted_history = []
        for move_info in move_history:
            if "move_san" in move_info:
                formatted_history.append({"san": move_info["move_san"]})
        return formatted_history

    def close(self):
        """Clean up resources"""
        if hasattr(self, 'engine') and self.engine:
            try:
                self.engine.quit()
            except Exception as e:
                logger.warning(f"Error while closing Stockfish engine: {e}")
            self.engine = None