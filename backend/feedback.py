# feedback.py
from flask import jsonify
import chess
import chess.engine
import time
import os
import random
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from collections import Counter

class ChessFeedbackAgent:
    """Agent that provides personalized feedback on a chess game."""
    
    def __init__(self, 
                 stockfish_path: str,
                 gemini_api_key: str,
                 stockfish_depth: int = 14,
                 temperature: float = 0.2,
                 feedback_tones: List[str] = None):
        """
        Initialize the chess feedback agent.
        
        Args:
            stockfish_path: Path to the Stockfish engine executable
            gemini_api_key: API key for Gemini
            stockfish_depth: Depth for Stockfish analysis
            temperature: Temperature for AI responses
            feedback_tones: List of feedback tones to choose from
        """
        self.stockfish_path = stockfish_path
        self.gemini_api_key = gemini_api_key
        self.stockfish_depth = stockfish_depth
        self.temperature = temperature
        
        # Initialize the Gemini API
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_api_key)
            self.genai = genai
            self.model = genai.GenerativeModel('gemini-pro', generation_config={
                'temperature': temperature,
                'max_output_tokens': 2048,
            })
        except ImportError:
            print("Please install the Google Generative AI package: pip install google-generativeai")
            raise
        
        # Default feedback tones if not provided
        self.feedback_tones = feedback_tones or [
            "Encouraging",
            "Critical",
            "Analytical",
            "Instructive",
            "Casual"
        ]
        
        # Counter for Q&A interactions
        self.qa_count = 0
        self.max_qa_count = 10
        
        # Store the game data for follow-up questions
        self.current_game_data = None
        
    def _get_stockfish_evaluation(self, fen: str) -> Dict[str, Any]:
        """
        Get Stockfish evaluation for a position.
        
        Args:
            fen: FEN string for the position
            
        Returns:
            Dictionary with evaluation information
        """
        try:
            engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
            engine.configure({"Threads": 4})
            
            board = chess.Board(fen)
            info = engine.analyse(board, chess.engine.Limit(depth=self.stockfish_depth))
            
            score = info["score"].white().score(mate_score=10000)
            if score is None:  # Handle mate scores
                mate = info["score"].white().mate()
                if mate is not None:
                    score = 10000 if mate > 0 else -10000
                else:
                    score = 0
                    
            engine.quit()
            
            return {
                "score": score,
                "pov_score": score if board.turn == chess.WHITE else -score,
                "mate": info["score"].white().mate()
            }
        except Exception as e:
            print(f"Error in Stockfish evaluation: {e}")
            return {"error": str(e)}
    
    def _classify_move_quality(self, eval_before: float, eval_after: float, player_color: chess.Color) -> str:
        """
        Classify the quality of a move based on the evaluation change.
        
        Args:
            eval_before: Evaluation before the move
            eval_after: Evaluation after the move
            player_color: Player's color (WHITE or BLACK)
            
        Returns:
            String classification of the move quality
        """
        # Convert to player's perspective
        if player_color == chess.BLACK:
            eval_before = -eval_before
            eval_after = -eval_after
        
        eval_diff = eval_after - eval_before
        
        if eval_diff < -300:
            return "Blunder"
        elif eval_diff < -150:
            return "Mistake"
        elif eval_diff < -75:
            return "Inaccuracy"
        elif eval_diff > 150:
            return "Excellent"
        elif eval_diff > 75:
            return "Good"
        else:
            return "Normal"
            
    def _analyze_game_phases(self, move_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze performance across different game phases.
        
        Args:
            move_history: List of moves with their analysis
            
        Returns:
            Dictionary with phase analysis
        """
        # Simple heuristic for phases
        opening_moves = min(15, len(move_history) // 3)
        endgame_index = max(0, len(move_history) - max(10, len(move_history) // 3))
        
        # Count move quality for different phases
        phases = {
            "opening": {"moves": 0, "blunders": 0, "mistakes": 0, "inaccuracies": 0, "good": 0, "excellent": 0},
            "middlegame": {"moves": 0, "blunders": 0, "mistakes": 0, "inaccuracies": 0, "good": 0, "excellent": 0},
            "endgame": {"moves": 0, "blunders": 0, "mistakes": 0, "inaccuracies": 0, "good": 0, "excellent": 0}
        }
        
        for i, move in enumerate(move_history):
            if i < opening_moves:
                phase = "opening"
            elif i >= endgame_index:
                phase = "endgame"
            else:
                phase = "middlegame"
                
            if "analysis" in move and "classification" in move["analysis"]:
                phases[phase]["moves"] += 1
                classification = move["analysis"]["classification"].lower()
                
                if "blunder" in classification:
                    phases[phase]["blunders"] += 1
                elif "mistake" in classification:
                    phases[phase]["mistakes"] += 1
                elif "inaccuracy" in classification:
                    phases[phase]["inaccuracies"] += 1
                elif "good" in classification:
                    phases[phase]["good"] += 1
                elif "excellent" in classification:
                    phases[phase]["excellent"] += 1
        
        return phases
    
    def _identify_patterns(self, move_history: List[Dict[str, Any]]) -> List[str]:
        """
        Identify patterns in the player's game.
        
        Args:
            move_history: List of moves with their analysis
            
        Returns:
            List of pattern descriptions
        """
        patterns = []
        
        # Count move classifications
        classifications = Counter([move.get("analysis", {}).get("classification", "Normal") 
                                for move in move_history])
        
        # Check for tendency to blunder
        blunder_count = classifications.get("Blunder", 0)
        if blunder_count >= len(move_history) / 8:
            patterns.append("Tendency to make blunders")
        
        # Check for conservative play
        normal_count = classifications.get("Normal", 0)
        if normal_count >= len(move_history) * 0.6:
            patterns.append("Conservative play style (many normal moves)")
        
        # Check for strong play
        good_excellent_count = classifications.get("Good", 0) + classifications.get("Excellent", 0)
        if good_excellent_count >= len(move_history) * 0.4:
            patterns.append("Strong play (many good/excellent moves)")
            
        # Check position evaluation trends
        if len(move_history) >= 6:
            # Check for improvement/decline in the later stages
            early_evals = [float(move.get("analysis", {}).get("stockfish_quality", "0").split()[0]) 
                           for move in move_history[:len(move_history)//2] 
                           if move.get("analysis", {}).get("stockfish_quality", "")]
            late_evals = [float(move.get("analysis", {}).get("stockfish_quality", "0").split()[0]) 
                          for move in move_history[len(move_history)//2:] 
                          if move.get("analysis", {}).get("stockfish_quality", "")]
            
            if early_evals and late_evals:
                avg_early = sum(early_evals) / len(early_evals)
                avg_late = sum(late_evals) / len(late_evals)
                
                if avg_late - avg_early > 0.5:
                    patterns.append("Improvement in play as the game progressed")
                elif avg_early - avg_late > 0.5:
                    patterns.append("Decline in play as the game progressed")
        
        return patterns
    
    def _check_suggestion_patterns(self, game_data: Dict[str, Any]) -> List[str]:
        """
        Analyze patterns in suggestion requests.
        
        Args:
            game_data: Dictionary with game information
            
        Returns:
            List of suggestion-related patterns
        """
        patterns = []
        
        # Check for suggestion requests
        suggestion_requests = game_data.get("suggestion_requests", [])
        if not suggestion_requests:
            return ["No suggestion requests made during the game"]
        
        # Check if suggestions were requested more in certain phases
        opening_requests = sum(1 for req in suggestion_requests if req.get("move_number", 0) <= 10)
        middlegame_requests = sum(1 for req in suggestion_requests 
                                if 10 < req.get("move_number", 0) <= 30)
        endgame_requests = sum(1 for req in suggestion_requests if req.get("move_number", 0) > 30)
        
        total_requests = len(suggestion_requests)
        
        if opening_requests > total_requests * 0.5:
            patterns.append("Frequently requested suggestions during the opening phase")
        if middlegame_requests > total_requests * 0.5:
            patterns.append("Frequently requested suggestions during the middlegame")
        if endgame_requests > total_requests * 0.5:
            patterns.append("Frequently requested suggestions during the endgame")
            
        # Check if suggestions were followed
        suggestions_followed = sum(1 for req in suggestion_requests if req.get("followed", False))
        if suggestions_followed > total_requests * 0.7:
            patterns.append("Frequently followed the suggested moves")
        elif suggestions_followed < total_requests * 0.3:
            patterns.append("Rarely followed the suggested moves")
            
        return patterns
    
    def _format_game_data_for_llm(self, game_data: Dict[str, Any]) -> str:
        """
        Format game data for LLM consumption.
        
        Args:
            game_data: Dictionary with game information
            
        Returns:
            Formatted string with game data
        """
        result = f"Game Result: {game_data['result']}\n\n"
        result += "Move History:\n"
        
        for i, move in enumerate(game_data["move_history"]):
            move_number = i // 2 + 1
            if i % 2 == 0:  # White's move
                result += f"{move_number}. {move.get('move_san', 'unknown')} "
            else:  # Black's move
                result += f"{move.get('move_san', 'unknown')} "
                
                # Add evaluation after black's move
                if "analysis" in move and "classification" in move["analysis"]:
                    result += f"[{move['analysis']['classification']}] "
                    
                result += "\n"
        
        result += "\n\nPhase Analysis:\n"
        for phase, stats in game_data["phase_analysis"].items():
            result += f"{phase.capitalize()}:\n"
            total_moves = stats["moves"]
            if total_moves > 0:
                result += f"- Moves: {total_moves}\n"
                result += f"- Blunders: {stats['blunders']} ({stats['blunders']/total_moves*100:.1f}%)\n"
                result += f"- Mistakes: {stats['mistakes']} ({stats['mistakes']/total_moves*100:.1f}%)\n"
                result += f"- Inaccuracies: {stats['inaccuracies']} ({stats['inaccuracies']/total_moves*100:.1f}%)\n"
                result += f"- Good moves: {stats['good']} ({stats['good']/total_moves*100:.1f}%)\n"
                result += f"- Excellent moves: {stats['excellent']} ({stats['excellent']/total_moves*100:.1f}%)\n"
            result += "\n"
        
        result += "Patterns Identified:\n"
        for pattern in game_data["patterns"]:
            result += f"- {pattern}\n"
        
        result += "\nSuggestion Patterns:\n"
        for pattern in game_data["suggestion_patterns"]:
            result += f"- {pattern}\n"
        
        return result
    
    def generate_feedback(self, 
                         move_history: List[Dict[str, Any]], 
                         game_result: str,
                         suggestion_requests: List[Dict[str, Any]] = None,
                         feedback_tone: str = None) -> Dict[str, Any]:
        """
        Generate comprehensive feedback on a chess game.
        
        Args:
            move_history: List of moves with their analysis
            game_result: Result of the game (win/loss/draw)
            suggestion_requests: List of suggestion requests during the game
            feedback_tone: Tone to use for feedback
            
        Returns:
            Dictionary with feedback information
        """
        # Reset Q&A counter for new feedback session
        self.qa_count = 0
        
        # Validate feedback tone
        if feedback_tone and feedback_tone not in self.feedback_tones:
            feedback_tone = self.feedback_tones[0]
        elif not feedback_tone:
            feedback_tone = self.feedback_tones[0]
            
        # Phase analysis
        phase_analysis = self._analyze_game_phases(move_history)
        
        # Pattern identification
        patterns = self._identify_patterns(move_history)
        
        # Suggestion pattern analysis
        game_data = {
            "move_history": move_history,
            "suggestion_requests": suggestion_requests or [],
            "result": game_result,
            "phase_analysis": phase_analysis,
            "patterns": patterns
        }
        
        suggestion_patterns = self._check_suggestion_patterns(game_data)
        game_data["suggestion_patterns"] = suggestion_patterns
        
        # Store the game data for follow-up questions
        self.current_game_data = game_data
        
        # Format data for LLM
        formatted_data = self._format_game_data_for_llm(game_data)
        
        # Generate feedback with Gemini
        prompt = f"""
        You are a chess coach providing feedback on a game. Below is the data about the game.
        
        {formatted_data}
        
        Please provide comprehensive feedback on the player's performance with the following elements:
        1. A brief summary of the game outcome
        2. Strengths demonstrated in the game
        3. Areas for improvement
        4. Specific patterns observed in their play
        5. Advice for future games
        
        Your feedback should be in a {feedback_tone} tone.
        """
        
        try:
            response = self.model.generate_content(prompt)
            feedback_text = response.text
            
            # Create a structured response
            response = {
                "feedback": feedback_text,
                "summary": {
                    "result": game_result,
                    "phase_analysis": phase_analysis,
                    "patterns": patterns,
                    "suggestion_patterns": suggestion_patterns
                },
                "tone": feedback_tone,
                "available_tones": self.feedback_tones,
                "remaining_questions": self.max_qa_count - self.qa_count
            }
            
            return response
            
        except Exception as e:
            print(f"Error generating feedback: {e}")
            return {
                "error": f"Failed to generate feedback: {str(e)}",
                "summary": {
                    "result": game_result,
                    "phase_analysis": phase_analysis,
                    "patterns": patterns,
                    "suggestion_patterns": suggestion_patterns
                }
            }
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Answer a follow-up question about the game.
        
        Args:
            question: Question from the player
            
        Returns:
            Dictionary with the answer
        """
        # Check if maximum questions reached
        if self.qa_count >= self.max_qa_count:
            return {
                "answer": "You've reached the maximum number of questions for this feedback session.",
                "remaining_questions": 0
            }
        
        # Check if we have game data
        if not self.current_game_data:
            return {
                "answer": "No game data available. Please analyze a game first.",
                "remaining_questions": self.max_qa_count - self.qa_count
            }
        
        # Increment question counter
        self.qa_count += 1
        
        # Format game data for context
        formatted_data = self._format_game_data_for_llm(self.current_game_data)
        
        # Generate answer with Gemini
        prompt = f"""
        You are a chess coach answering a question about a chess game. Below is the data about the game:
        
        {formatted_data}
        
        The player's question is: "{question}"
        
        Please provide a helpful, accurate answer based on the game data. If the question requires specific move analysis that's not available in the data, mention that limitation. Be specific and use any relevant information from the game data.
        """
        
        try:
            response = self.model.generate_content(prompt)
            answer_text = response.text
            
            return {
                "answer": answer_text,
                "remaining_questions": self.max_qa_count - self.qa_count
            }
            
        except Exception as e:
            print(f"Error answering question: {e}")
            return {
                "error": f"Failed to answer question: {str(e)}",
                "remaining_questions": self.max_qa_count - self.qa_count
            }
    
    def get_available_tones(self) -> List[str]:
        """Get available feedback tones."""
        return self.feedback_tones
    
    def change_feedback_tone(self, 
                            move_history: List[Dict[str, Any]], 
                            game_result: str,
                            suggestion_requests: List[Dict[str, Any]] = None,
                            new_tone: str = None) -> Dict[str, Any]:
        """
        Change the tone of the feedback.
        
        Args:
            move_history: List of moves with their analysis
            game_result: Result of the game (win/loss/draw)
            suggestion_requests: List of suggestion requests during the game
            new_tone: New tone to use for feedback
            
        Returns:
            Dictionary with feedback in the new tone
        """
        # Validate tone
        if new_tone not in self.feedback_tones:
            return {
                "error": f"Invalid tone. Available tones: {', '.join(self.feedback_tones)}",
                "available_tones": self.feedback_tones
            }
            
        # Generate new feedback with the specified tone
        return self.generate_feedback(move_history, game_result, suggestion_requests, new_tone)
    
    def invoke_debate(self, 
                     debate_agent, 
                     move_or_position: str,
                     is_move: bool = False,
                     fen: str = None,
                     num_perspectives: int = 2) -> Dict[str, Any]:
        """
        Invoke the debate agent for a deeper analysis.
        
        Args:
            debate_agent: ChessDebateAgent instance
            move_or_position: Move or position to debate
            is_move: Whether the input is a move (True) or a position (False)
            fen: FEN string for the position if is_move is True
            num_perspectives: Number of perspectives to include
            
        Returns:
            Dictionary with debate results
        """
        if is_move and fen:
            # Debate a specific move
            return debate_agent.analyze_move_debate(fen, move_or_position, num_perspectives)
        else:
            # Debate a position
            position = move_or_position if not is_move else fen
            return debate_agent.generate_debate(position, num_perspectives)