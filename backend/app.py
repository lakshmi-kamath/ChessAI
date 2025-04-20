# app.py
from flask import Flask, render_template, request, jsonify
import chess
import json
import os
from Retrieval import ChessRetrievalAgent
from selector import ChessSelectionAgent
from analyzer import HumanMoveAnalysisAgent
from debater import ChessDebateAgent
from engine_manager import ChessEngineManager

app = Flask(__name__)

# Initialize the engine manager first (singleton)
stockfish_path = "/opt/homebrew/bin/stockfish"  # Path to Stockfish engine
engine_manager = ChessEngineManager(stockfish_path)  # Ensure engine path is passed during initialization

# Initialize agents (adjust paths as needed)
retrieval_agent = ChessRetrievalAgent(
        index_path="/Users/lakshmikamath/Desktop/tdl/knowledge-base/chess_positions.index",
        embeddings_path="/Users/lakshmikamath/Desktop/tdl/knowledge-base/raw_positions.pkl",
        metadata_path="/Users/lakshmikamath/Desktop/tdl/knowledge-base/positions_metadata.pkl",
        model_name="all-MiniLM-L6-v2",
        top_k=5
    )

selection_agent = ChessSelectionAgent(
    retrieval_agent=retrieval_agent,
    engine_manager=engine_manager,  # Pass engine manager instead of path
    gemini_api_key="AIzaSyBZhEyYLeNGqCrxCLIAOxUeXBdCYW5Kly4",
    stockfish_depth=14
)

# Initialize the human move analysis agent
human_move_agent = HumanMoveAnalysisAgent(
    engine_manager=engine_manager,  # Pass engine manager instead of path
    gemini_api_key="AIzaSyBZhEyYLeNGqCrxCLIAOxUeXBdCYW5Kly4",
    stockfish_depth=14
)

# Initialize the debate agent with different chess styles
debate_agent = ChessDebateAgent(
    engine_manager=engine_manager,  # Pass engine manager instead of path
    gemini_api_key="AIzaSyBZhEyYLeNGqCrxCLIAOxUeXBdCYW5Kly4",
    stockfish_depth=14,
    debate_styles=[
        "Classical",
        "Tactical",
        "Hypermodern",
        "Positional"
    ]
)


# Game state (simple in-memory storage)
active_games = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/new_game', methods=['POST'])
def new_game():
    game_id = str(len(active_games) + 1)  # Simple ID generation
    active_games[game_id] = {
        'board': chess.Board(),
        'history': [],
        'debates': [],  # Array to store debate history
        'suggestion_requests': []  # Array to store suggestion requests
    }
    
    # Return reset_debate flag to clear debate section on frontend
    return jsonify({
        'game_id': game_id,
        'fen': active_games[game_id]['board'].fen(),
        'reset_debate': True,  # Signal to the frontend to clear debate section
        'clear_debates': True   # Added an additional flag for clarity
    })

@app.route('/api/make_move', methods=['POST'])
def make_move():
    data = request.json
    game_id = data.get('game_id')
    move = data.get('move')
    
    if game_id not in active_games:
        return jsonify({'error': 'Invalid game ID'}), 400
    
    game = active_games[game_id]
    board = game['board']
    
    # Parse and validate move
    try:
        move_obj = chess.Move.from_uci(move)
        if move_obj not in board.legal_moves:
            return jsonify({'error': 'Illegal move'}), 400

        # Get current position before making the move
        current_fen = board.fen()
        
        # Get SAN notation before making the move
        move_san = board.san(move_obj)
        
        # Analyze the human move using HumanMoveAnalysisAgent
        human_move_analysis = human_move_agent.analyze_human_move(current_fen, move, game['history'])
        
        # Make the human move
        board.push(move_obj)
        
        # Record move in history with analysis
        move_number = len(game['history']) // 2 + 1
        game['history'].append({
            'move': move,
            'move_san': move_san,  # We calculated this before pushing the move
            'fen': current_fen,
            'analysis': human_move_analysis,  # Store the analysis in the history
            'move_number': move_number,
            'player': 'white' if len(game['history']) % 2 == 0 else 'black'
        })
        
        # Make bot move if the game is not over
        bot_move = None
        bot_move_san = None
        if not board.is_game_over():
            # Use Stockfish to get the best move for the bot
            stockfish_moves = selection_agent.get_stockfish_top_moves(board.fen(), 1)
            if stockfish_moves:
                bot_move = stockfish_moves[0]['move']
                bot_move_obj = chess.Move.from_uci(bot_move)
                bot_move_san = board.san(bot_move_obj)  # Get SAN before making the move
                board.push(bot_move_obj)
        
        # Check game status
        game_status = {
            'is_game_over': board.is_game_over(),
            'result': board.result() if board.is_game_over() else None,
            'is_check': board.is_check(),
            'is_checkmate': board.is_checkmate(),
            'is_stalemate': board.is_stalemate()
        }
        
        # For frontend compatibility, create empty suggestion
        empty_suggestion = {
            'gemini_analysis': '',
            'top_moves': []
        }
        
        # Format human move analysis for frontend
        formatted_human_move_analysis = {
            'classification': human_move_analysis.get('classification', ''),
            'explanation': human_move_analysis.get('explanation', ''),
            'stockfish_quality': human_move_analysis.get('stockfish_quality', ''),
            'llm_analysis': human_move_analysis.get('gemini_analysis', {}).get('explanation', '')
        }
        
        return jsonify({
            'fen': board.fen(),
            'human_move': move,
            'human_move_san': move_san,
            'bot_move': bot_move,
            'bot_move_san': bot_move_san,
            'suggestion': empty_suggestion,
            'human_move_analysis': formatted_human_move_analysis,
            'game_status': game_status
        })
        
    except ValueError as e:
        return jsonify({'error': f'Invalid move format: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Error processing move: {str(e)}'}), 500

@app.route('/api/get_suggestion', methods=['POST'])
def get_suggestion():
    data = request.json
    fen = data.get('fen')
    game_id = data.get('game_id')  # Added game_id parameter
    
    if not fen:
        return jsonify({'error': 'FEN position required'}), 400
    
    try:
        # Validate FEN
        board = chess.Board(fen)
        
        # Get suggestion - this is now only done when explicitly requested
        suggestion = selection_agent.suggest_move(fen)
        
        # Record suggestion request if game_id is provided
        if game_id and game_id in active_games:
            move_number = len(active_games[game_id]['history']) // 2 + 1
            active_games[game_id]['suggestion_requests'].append({
                'move_number': move_number,
                'fen': fen,
                'suggested_moves': suggestion.get('top_moves', []),
                'followed': False  # Will be updated if player follows this suggestion
            })
        
        return jsonify({
            'suggestion': {
                'gemini_analysis': suggestion.get('gemini_analysis', ''),
                'top_moves': suggestion.get('top_moves', [])[:3]
            }
        })
    except ValueError:
        return jsonify({'error': 'Invalid FEN position'}), 400

@app.route('/api/analyze_move', methods=['POST'])
def analyze_move():
    """Endpoint specifically for analyzing a human move without making it"""
    data = request.json
    fen = data.get('fen')
    move = data.get('move')
    game_id = data.get('game_id')
    
    if not fen or not move:
        return jsonify({'error': 'FEN position and move required'}), 400
    
    try:
        # Validate FEN
        board = chess.Board(fen)
        
        # Get move history if game_id is provided
        move_history = []
        if game_id and game_id in active_games:
            move_history = active_games[game_id]['history']
        
        # Analyze the move
        analysis = human_move_agent.analyze_human_move(fen, move, move_history)
        
        if "error" in analysis:
            return jsonify({'error': analysis["error"]}), 400
            
        return jsonify({
            'analysis': {
                'classification': analysis.get('classification', ''),
                'explanation': analysis.get('explanation', ''),
                'stockfish_quality': analysis.get('stockfish_quality', ''),
                'impact_assessment': analysis.get('impact_assessment', ''),
                'suggested_improvements': analysis.get('suggested_improvements', '')
            }
        })
    except ValueError:
        return jsonify({'error': 'Invalid FEN position or move'}), 400

# Routes for the debate functionality

@app.route('/api/get_position_debate', methods=['POST'])
def get_position_debate():
    """Endpoint for generating a debate about a position"""
    data = request.json
    fen = data.get('fen')
    game_id = data.get('game_id')
    num_perspectives = data.get('perspectives', 2)
    
    if not fen:
        return jsonify({'error': 'FEN position required'}), 400
    
    try:
        # Validate FEN
        board = chess.Board(fen)
        
        # Generate position debate
        debate_result = debate_agent.generate_debate(fen, num_perspectives)
        
        if "error" in debate_result:
            return jsonify({'error': debate_result["error"]}), 400
        
        # Store debate in game history if game_id provided    
        if game_id and game_id in active_games:
            active_games[game_id]['debates'].append({
                'type': 'position',
                'fen': fen,
                'result': debate_result
            })
            
        return jsonify({
            'debate': debate_result
        })
    except ValueError as e:
        return jsonify({'error': f'Invalid FEN position: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Error generating position debate: {str(e)}'}), 500

@app.route('/api/get_move_debate', methods=['POST'])
def get_move_debate():
    """Endpoint for generating a debate about a specific move"""
    data = request.json
    fen = data.get('fen')
    move = data.get('move')
    game_id = data.get('game_id')
    num_perspectives = data.get('perspectives', 2)
    
    if not fen or not move:
        return jsonify({'error': 'FEN position and move required'}), 400
    
    try:
        # Validate FEN and move
        board = chess.Board(fen)
        move_obj = chess.Move.from_uci(move)
        
        # Check if move is legal
        if move_obj not in board.legal_moves:
            return jsonify({'error': 'Illegal move for the given position'}), 400
        
        # Generate move debate
        debate_result = debate_agent.analyze_move_debate(fen, move, num_perspectives)
        
        if "error" in debate_result:
            return jsonify({'error': debate_result["error"]}), 400
        
        # Store debate in game history if game_id provided    
        if game_id and game_id in active_games:
            active_games[game_id]['debates'].append({
                'type': 'move',
                'fen': fen,
                'move': move,
                'result': debate_result
            })
            
        return jsonify({
            'debate': debate_result
        })
    except ValueError as e:
        return jsonify({'error': f'Invalid FEN position or move format: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Error generating move debate: {str(e)}'}), 500

# Route to get available debate styles
@app.route('/api/get_debate_styles', methods=['GET'])
def get_debate_styles():
    """Endpoint for retrieving available debate styles"""
    return jsonify({
        'styles': debate_agent.debate_styles
    })

@app.teardown_appcontext
def cleanup(exception=None):
    """Close resources when the application shuts down."""
    engine_manager.close()

if __name__ == '__main__':
    try:
        app.run(debug=True)
    finally:
        # Ensure engine is closed even if app crashes
        engine_manager.close()