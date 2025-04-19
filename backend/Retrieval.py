import faiss
import numpy as np
import chess
import chess.pgn
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional, Union
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ChessRetrievalAgent")

class ChessRetrievalAgent:
    """
    Chess Retrieval Agent that uses a vector database
    to retrieve relevant chess knowledge based on board positions.
    """
    
    def __init__(
        self,
        index_path: str,
        embeddings_path: str,
        metadata_path: str,
        model_name: str = "all-MiniLM-L6-v2",
        top_k: int = 5,
        reranking_threshold: float = 0.75,
    ):
        """
        Initialize the Chess Retrieval Agent.
        
        Args:
            index_path: Path to the FAISS index file
            embeddings_path: Path to stored embeddings (pickle file)
            metadata_path: Path to metadata associated with the vectors (pickle file)
            model_name: SentenceTransformer model name
            top_k: Number of relevant chunks to retrieve
            reranking_threshold: Threshold for reranking results
        """
        self.top_k = top_k
        self.reranking_threshold = reranking_threshold
        
        # Load the embedding model
        logger.info(f"Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        
        # Load FAISS index
        logger.info(f"Loading FAISS index from {index_path}")
        self.index = faiss.read_index(index_path)
        
        # Load metadata from pickle file
        logger.info(f"Loading metadata from {metadata_path}")
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        # Load raw positions/embeddings if needed
        logger.info(f"Loading raw positions data from {embeddings_path}")
        with open(embeddings_path, 'rb') as f:
            self.raw_positions = pickle.load(f)
            
        logger.info("Chess Retrieval Agent initialized successfully")
    
    def embed_fen(self, fen: str) -> np.ndarray:
        """
        Create an embedding for a FEN position.
        
        Args:
            fen: FEN notation for a chess position
        
        Returns:
            Embedding vector for the FEN
        """
        # Create a textual representation of the board for embedding
        board = chess.Board(fen)
        
        # Create a rich textual description of the position
        description = self._create_position_description(board)
        
        # Generate embedding
        embedding = self.embedding_model.encode(description)
        return embedding.reshape(1, -1).astype('float32')
    
    def _create_position_description(self, board: chess.Board) -> str:
        """
        Create a rich textual description of a chess position for better embedding.
        
        Args:
            board: Chess board object
        
        Returns:
            Textual description of the position
        """
        # Basic position information
        fen = board.fen()
        turn = "White" if board.turn else "Black"
        
        # Piece placement analysis
        white_pieces = {
            'P': list(board.pieces(chess.PAWN, chess.WHITE)),
            'N': list(board.pieces(chess.KNIGHT, chess.WHITE)),
            'B': list(board.pieces(chess.BISHOP, chess.WHITE)),
            'R': list(board.pieces(chess.ROOK, chess.WHITE)),
            'Q': list(board.pieces(chess.QUEEN, chess.WHITE)),
            'K': list(board.pieces(chess.KING, chess.WHITE))
        }
        
        black_pieces = {
            'p': list(board.pieces(chess.PAWN, chess.BLACK)),
            'n': list(board.pieces(chess.KNIGHT, chess.BLACK)),
            'b': list(board.pieces(chess.BISHOP, chess.BLACK)),
            'r': list(board.pieces(chess.ROOK, chess.BLACK)),
            'q': list(board.pieces(chess.QUEEN, chess.BLACK)),
            'k': list(board.pieces(chess.KING, chess.BLACK))
        }
        
        # Create textual description
        description = f"FEN: {fen}. {turn} to move. "
        description += f"Material balance: White has {len(white_pieces['P'])} pawns, "
        description += f"{len(white_pieces['N'])} knights, {len(white_pieces['B'])} bishops, "
        description += f"{len(white_pieces['R'])} rooks, {len(white_pieces['Q'])} queens. "
        description += f"Black has {len(black_pieces['p'])} pawns, {len(black_pieces['n'])} knights, "
        description += f"{len(black_pieces['b'])} bishops, {len(black_pieces['r'])} rooks, "
        description += f"{len(black_pieces['q'])} queens. "
        
        # Add information about king safety and pawn structure
        description += self._analyze_king_safety(board)
        description += self._analyze_pawn_structure(board)
        
        # Add additional position context
        description += self._analyze_position_context(board)
        
        return description
    
    def _analyze_king_safety(self, board: chess.Board) -> str:
        """Analyze king safety for both sides."""
        white_king_sq = board.king(chess.WHITE)
        black_king_sq = board.king(chess.BLACK)
        
        white_king_file = chess.square_file(white_king_sq)
        white_king_rank = chess.square_rank(white_king_sq)
        black_king_file = chess.square_file(black_king_sq)
        black_king_rank = chess.square_rank(black_king_sq)
        
        # Simple king safety metric based on position
        w_safety = "castled kingside" if white_king_file >= 5 and white_king_rank == 0 else \
                  "castled queenside" if white_king_file <= 2 and white_king_rank == 0 else \
                  "center" if 2 < white_king_file < 5 else "uncastled"
                  
        b_safety = "castled kingside" if black_king_file >= 5 and black_king_rank == 7 else \
                  "castled queenside" if black_king_file <= 2 and black_king_rank == 7 else \
                  "center" if 2 < black_king_file < 5 else "uncastled"
        
        return f"White king is {w_safety}. Black king is {b_safety}. "
    
    def _analyze_pawn_structure(self, board: chess.Board) -> str:
        """Analyze pawn structure."""
        # Simple pawn structure analysis
        white_pawns = list(board.pieces(chess.PAWN, chess.WHITE))
        black_pawns = list(board.pieces(chess.PAWN, chess.BLACK))
        
        # Count pawns on each file
        white_pawn_files = [chess.square_file(sq) for sq in white_pawns]
        black_pawn_files = [chess.square_file(sq) for sq in black_pawns]
        
        # Check for doubled pawns
        white_doubled = any(white_pawn_files.count(f) > 1 for f in range(8))
        black_doubled = any(black_pawn_files.count(f) > 1 for f in range(8))
        
        pawn_str = "Pawn structure: "
        if white_doubled:
            pawn_str += "White has doubled pawns. "
        if black_doubled:
            pawn_str += "Black has doubled pawns. "
        if not white_doubled and not black_doubled:
            pawn_str += "No doubled pawns. "
            
        return pawn_str
    
    def _analyze_position_context(self, board: chess.Board) -> str:
        """Analyze additional context about the position."""
        # Game phase detection
        total_pieces = sum(1 for _ in board.pieces(chess.PAWN, chess.WHITE))
        total_pieces += sum(1 for _ in board.pieces(chess.PAWN, chess.BLACK))
        total_pieces += sum(1 for _ in board.pieces(chess.KNIGHT, chess.WHITE))
        total_pieces += sum(1 for _ in board.pieces(chess.KNIGHT, chess.BLACK))
        total_pieces += sum(1 for _ in board.pieces(chess.BISHOP, chess.WHITE))
        total_pieces += sum(1 for _ in board.pieces(chess.BISHOP, chess.BLACK))
        total_pieces += sum(1 for _ in board.pieces(chess.ROOK, chess.WHITE))
        total_pieces += sum(1 for _ in board.pieces(chess.ROOK, chess.BLACK))
        total_pieces += sum(1 for _ in board.pieces(chess.QUEEN, chess.WHITE))
        total_pieces += sum(1 for _ in board.pieces(chess.QUEEN, chess.BLACK))
        total_pieces += sum(1 for _ in board.pieces(chess.KING, chess.WHITE))
        total_pieces += sum(1 for _ in board.pieces(chess.KING, chess.BLACK))
        
        # Determine game phase
        if total_pieces >= 28:
            phase = "opening"
        elif total_pieces >= 15:
            phase = "middlegame"
        else:
            phase = "endgame"
            
        # Check for common openings or structures
        # Just a simplified example - real implementation would be more robust
        fen_parts = board.fen().split()
        halfmove = int(fen_parts[5]) if len(fen_parts) > 5 else 0
        fullmove = int(fen_parts[5]) if len(fen_parts) > 5 else 1
        
        opening_info = ""
        if fullmove <= 5:
            center_pawns = 0
            if board.piece_at(chess.E4) and board.piece_at(chess.E4).piece_type == chess.PAWN:
                center_pawns += 1
            if board.piece_at(chess.D4) and board.piece_at(chess.D4).piece_type == chess.PAWN:
                center_pawns += 1
            if board.piece_at(chess.E5) and board.piece_at(chess.E5).piece_type == chess.PAWN:
                center_pawns += 1
            if board.piece_at(chess.D5) and board.piece_at(chess.D5).piece_type == chess.PAWN:
                center_pawns += 1
                
            if center_pawns >= 2:
                opening_info = "Central pawn structure typical of open games. "
            
        return f"Position appears to be in the {phase}. {opening_info}"
    
    def retrieve_knowledge(self, fen: str) -> List[Dict]:
        """
        Retrieve relevant knowledge from the database based on FEN.
        
        Args:
            fen: FEN notation for a chess position
        
        Returns:
            List of retrieved chunks with metadata
        """
        logger.info(f"Retrieving knowledge for position: {fen}")
        
        # Generate embedding for the query
        query_embedding = self.embed_fen(fen)
        
        # Search the index
        distances, indices = self.index.search(query_embedding, self.top_k)
        
        # Gather results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # Valid index
                # Handle different metadata structures
                if isinstance(self.metadata, list):
                    metadata = self.metadata[idx] if idx < len(self.metadata) else {}
                elif isinstance(self.metadata, dict):
                    metadata = self.metadata.get(idx, {})
                else:
                    # Try to access it as an attribute or key
                    try:
                        metadata = self.metadata[idx]
                    except:
                        logger.warning(f"Could not access metadata for index {idx}")
                        metadata = {}
                
                score = 1.0 - distances[0][i]  # Convert distance to similarity
                
                # Extract content and moves based on metadata structure
                content = ""
                moves = []
                evaluation = 0.0
                
                # Try to extract content from different potential metadata formats
                if isinstance(metadata, dict):
                    content = metadata.get("content", "")
                    moves = metadata.get("moves", [])
                    evaluation = metadata.get("evaluation", 0.0)
                elif hasattr(metadata, "content"):
                    content = metadata.content
                    moves = getattr(metadata, "moves", [])
                    evaluation = getattr(metadata, "evaluation", 0.0)
                
                results.append({
                    "chunk_id": idx,
                    "score": float(score),
                    "content": content,
                    "moves": moves,
                    "evaluation": evaluation,
                    "metadata": metadata
                })
        
        # Apply advanced reranking
        reranked_results = self._rerank_results(results, fen)
        
        logger.info(f"Retrieved {len(reranked_results)} relevant chunks")
        return reranked_results
    
    def _rerank_results(self, results: List[Dict], fen: str) -> List[Dict]:
        """
        Rerank the retrieved results using additional chess-specific heuristics.
        
        Args:
            results: Initial retrieval results
            fen: FEN notation of the query
        
        Returns:
            Reranked results
        """
        # Parse the current board
        current_board = chess.Board(fen)
        
        # Calculate similarity of piece placement and position dynamics
        for result in results:
            # Try to extract FEN from metadata
            result_fen = ""
            metadata = result.get("metadata", {})
            
            # Try different ways to access FEN in metadata
            if isinstance(metadata, dict):
                result_fen = metadata.get("fen", "")
            elif hasattr(metadata, "fen"):
                result_fen = metadata.fen
            
            # If we found a FEN, calculate position similarity
            if result_fen:
                # Calculate position similarity score
                pos_similarity = self._calculate_position_similarity(current_board, result_fen)
                
                # Blend the scores (weighted average)
                # 60% vector similarity, 40% position similarity
                result["score"] = 0.6 * result["score"] + 0.4 * pos_similarity
        
        # Sort by new score
        reranked_results = sorted(results, key=lambda x: x["score"], reverse=True)
        
        # Filter by threshold
        filtered_results = [r for r in reranked_results if r["score"] >= self.reranking_threshold]
        
        # Return at least one result even if below threshold
        return filtered_results if filtered_results else reranked_results[:1]
    
    def _calculate_position_similarity(self, board: chess.Board, other_fen: str) -> float:
        """
        Calculate similarity between two chess positions.
        
        Args:
            board: Current board position
            other_fen: FEN of the position to compare with
        
        Returns:
            Similarity score (0.0 to 1.0)
        """
        try:
            other_board = chess.Board(other_fen)
            
            # Compare piece placement (basic approach)
            similarity = 0.0
            max_pieces = 32  # Maximum number of pieces on a board
            
            # Count matching pieces
            matching_pieces = 0
            for square in chess.SQUARES:
                piece1 = board.piece_at(square)
                piece2 = other_board.piece_at(square)
                
                if piece1 is None and piece2 is None:
                    matching_pieces += 0.5  # Empty squares match but count less
                elif piece1 is not None and piece2 is not None and piece1 == piece2:
                    matching_pieces += 1.0  # Same piece on same square
            
            # Normalize to 0.0-1.0 range
            similarity = matching_pieces / max_pieces
            return similarity
            
        except Exception as e:
            logger.warning(f"Error calculating position similarity: {e}")
            return 0.0
    
    def analyze_position_context(self, fen: str) -> Dict[str, str]:
        """
        Generate rich contextual information about the position for LLM context.
        
        Args:
            fen: FEN notation of the position
            
        Returns:
            Dictionary with contextual information
        """
        board = chess.Board(fen)
        
        # Create context dictionary
        context = {
            "fen": fen,
            "turn": "White" if board.turn else "Black",
            "full_moves": board.fullmove_number,
            "half_moves": board.halfmove_clock,
            "game_phase": self._determine_game_phase(board),
            "position_description": self._create_position_description(board),
            "material_advantage": self._analyze_material_advantage(board),
            "tactical_themes": self._identify_tactical_themes(board)
        }
        
        return context
    
    def _determine_game_phase(self, board: chess.Board) -> str:
        """Determine the phase of the game (opening, middlegame, endgame)"""
        # Count major pieces
        queens = len(list(board.pieces(chess.QUEEN, chess.WHITE))) + len(list(board.pieces(chess.QUEEN, chess.BLACK)))
        rooks = len(list(board.pieces(chess.ROOK, chess.WHITE))) + len(list(board.pieces(chess.ROOK, chess.BLACK)))
        minors = (len(list(board.pieces(chess.KNIGHT, chess.WHITE))) + len(list(board.pieces(chess.KNIGHT, chess.BLACK))) +
                  len(list(board.pieces(chess.BISHOP, chess.WHITE))) + len(list(board.pieces(chess.BISHOP, chess.BLACK))))
        
        # Simple heuristic for game phase
        if board.fullmove_number <= 10:
            return "opening"
        elif queens >= 1 and (rooks >= 3 or minors >= 4):
            return "middlegame"
        else:
            return "endgame"
    
    def _analyze_material_advantage(self, board: chess.Board) -> str:
        """Analyze material advantage"""
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
        
        diff = white_material - black_material
        
        if abs(diff) < 0.5:
            return "Material is balanced"
        elif diff > 0:
            return f"White has a material advantage of approximately {diff} pawns"
        else:
            return f"Black has a material advantage of approximately {-diff} pawns"
    
    def _identify_tactical_themes(self, board: chess.Board) -> List[str]:
        """Identify potential tactical themes in the position"""
        themes = []
        
        # Check for pins, forks, etc.
        # This is a simplified version - a real implementation would be more complex
        
        # Check for check
        if board.is_check():
            themes.append("check")
            
        # Check for discovered attack potential
        # This is just a placeholder - real implementation would be more sophisticated
        if len(list(board.legal_moves)) > 20:
            themes.append("potential tactics")
            
        return themes if themes else ["no obvious tactical themes"]

    
    