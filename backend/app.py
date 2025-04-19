# app.py
from flask import Flask, render_template, request, jsonify
import chess
import json
import os
from Retrieval import ChessRetrievalAgent
from selector import ChessSelectionAgent

app = Flask(__name__)

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
    stockfish_path="/opt/homebrew/bin/stockfish",  # Path to Stockfish engine
    gemini_api_key="your_api_key_here",  # Replace with your actual API key
    stockfish_depth=14
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
        'history': []
    }
    return jsonify({
        'game_id': game_id,
        'fen': active_games[game_id]['board'].fen()
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
    except ValueError:
        return jsonify({'error': 'Invalid move format'}), 400
    
    # Make the human move
    board.push(move_obj)
    
    # Get agent suggestion about the human move
    current_fen = board.fen()
    suggestion = selection_agent.suggest_move(current_fen, move)
    
    # Record move in history
    game['history'].append({
        'move': move,
        'fen': current_fen,
        'suggestion': suggestion
    })
    
    # Make bot move if the game is not over
    bot_move = None
    bot_move_san = None
    if not board.is_game_over():
        # Use Stockfish to get the best move for the bot
        stockfish_moves = selection_agent.get_stockfish_top_moves(board.fen(), 1)
        if stockfish_moves:
            bot_move = stockfish_moves[0]['move']
            bot_move_san = stockfish_moves[0]['san']
            bot_move_obj = chess.Move.from_uci(bot_move)
            board.push(bot_move_obj)
    
    # Check game status
    game_status = {
        'is_game_over': board.is_game_over(),
        'result': board.result() if board.is_game_over() else None,
        'is_check': board.is_check(),
        'is_checkmate': board.is_checkmate(),
        'is_stalemate': board.is_stalemate()
    }
    
    return jsonify({
        'fen': board.fen(),
        'human_move': move,
        'bot_move': bot_move,
        'bot_move_san': bot_move_san,
        'suggestion': {
            'gemini_analysis': suggestion.get('gemini_analysis', ''),
            'top_moves': suggestion.get('top_moves', [])[:3],
            'human_move_analysis': suggestion.get('human_move_analysis', {})
        },
        'game_status': game_status
    })

@app.route('/api/get_suggestion', methods=['POST'])
def get_suggestion():
    data = request.json
    fen = data.get('fen')
    
    if not fen:
        return jsonify({'error': 'FEN position required'}), 400
    
    try:
        # Validate FEN
        board = chess.Board(fen)
        
        # Get suggestion without human move
        suggestion = selection_agent.suggest_move(fen)
        
        return jsonify({
            'suggestion': {
                'gemini_analysis': suggestion.get('gemini_analysis', ''),
                'top_moves': suggestion.get('top_moves', [])[:3]
            }
        })
    except ValueError:
        return jsonify({'error': 'Invalid FEN position'}), 400

if __name__ == '__main__':
    app.run(debug=True)