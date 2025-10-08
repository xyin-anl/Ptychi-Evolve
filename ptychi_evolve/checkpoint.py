"""
Checkpoint management for discovery sessions.
Handles saving and loading discovery state for resume functionality.
"""

import json
import time
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, Optional
from .logging import get_logger


class CheckpointManager:
    """Manages checkpoints for discovery sessions to enable resume functionality."""
    
    def __init__(self, save_path: Path, name: str, checkpoint_interval: int = 10):
        """
        Initialize checkpoint manager.
        
        Args:
            save_path: Directory to save checkpoints
            name: Name of the discovery session
            checkpoint_interval: Save checkpoint every N algorithms (default: 10)
        """
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_file = self.save_path / f"{name}_checkpoint.json"
        self.backup_file = self.save_path / f"{name}_checkpoint.backup.json"
        
        # Initialize logger
        self.log = get_logger(__name__, verbose=True)  # Always verbose for checkpoints
        
    def save_checkpoint(self, state: Dict[str, Any], force: bool = False) -> bool:
        """
        Save checkpoint if interval reached or forced.
        
        Args:
            state: Discovery state to save
            force: Force save regardless of interval
        """
        total_generated = state.get('stats', {}).get('total_generated', 0)
        
        # Check if we should save
        if not force and total_generated % self.checkpoint_interval != 0:
            return False
            
        # Prepare checkpoint data
        checkpoint = {
            'timestamp': time.time(),
            'discovery_state': state,
            'version': '1.0'
        }
        
        try:
            # Backup existing checkpoint if it exists
            if self.checkpoint_file.exists():
                import shutil
                shutil.copy2(self.checkpoint_file, self.backup_file)
            
            # Write to temporary file first (atomic save)
            with tempfile.NamedTemporaryFile(mode='w', delete=False, 
                                           dir=self.save_path, 
                                           suffix='.tmp') as tmp_file:
                try:
                    json.dump(checkpoint, tmp_file, indent=2, allow_nan=False)
                    tmp_path = tmp_file.name
                except (TypeError, ValueError) as json_error:
                    self.log.error(f"JSON serialization error: {json_error}")
                    self.log.error(f"Problematic data structure - stats: {state.get('stats', {})}")
                    self.log.error(f"Number of algorithms: {len(state.get('history', {}).get('algorithms', []))}")
                    # Log which algorithms are being dropped to save a minimal checkpoint
                    try:
                        removed_ids = [a.get('id', 'unknown') for a in state.get('history', {}).get('algorithms', [])]
                        if removed_ids:
                            self.log.warning(f"Removed algorithms from checkpoint: {', '.join(map(str, removed_ids))}")
                    except Exception:
                        pass
                    # Try to save a minimal checkpoint
                    minimal_checkpoint = {
                        'timestamp': checkpoint['timestamp'],
                        'discovery_state': {
                            'stats': state.get('stats', {}),
                            'history': {
                                'algorithms': [],  # Empty to avoid serialization issues
                                'metadata': state.get('history', {}).get('metadata', {})
                            },
                            'experiment_context': str(state.get('experiment_context', ''))
                        },
                        'version': '1.0',
                        'error': f'Full checkpoint failed: {str(json_error)}'
                    }
                    tmp_file.seek(0)
                    tmp_file.truncate(0)
                    json.dump(minimal_checkpoint, tmp_file, indent=2, allow_nan=False)
                    tmp_path = tmp_file.name
                    self.log.warning("Saved minimal checkpoint due to serialization error")
            
            # Atomic rename to replace checkpoint file
            os.replace(tmp_path, self.checkpoint_file)
                
            # Log success - always show checkpoint saves
            self.log.checkpoint(f"✓ Checkpoint saved to {self.checkpoint_file}")
            self.log.checkpoint(f"  Generation: {total_generated}, Algorithms: {len(state.get('history', {}).get('algorithms', []))}")
            return True
        except Exception as e:
            # Clean up temp file if it exists
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            
            # Try to restore backup if save failed
            if self.backup_file.exists():
                import shutil
                shutil.copy2(self.backup_file, self.checkpoint_file)
            
            self.log.error(f"✗ Failed to save checkpoint: {e}")
            raise Exception(f"Failed to save checkpoint: {e}")
    
    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Load the most recent checkpoint.
        
        Returns:
            Discovery state if checkpoint exists, None otherwise
        """
        if not self.checkpoint_file.exists():
            self.log.info(f"No checkpoint found at {self.checkpoint_file}")
            return None
            
        try:
            with open(self.checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                
            # Log successful load
            state = checkpoint.get('discovery_state', {})
            stats = state.get('stats', {})
            self.log.checkpoint(f"✓ Checkpoint loaded from {self.checkpoint_file}")
            self.log.checkpoint(f"  Generation: {stats.get('total_generated', 0)}, Timestamp: {time.ctime(checkpoint.get('timestamp', 0))}")
            return state
            
        except Exception as e:
            self.log.error(f"✗ Failed to load main checkpoint: {e}")
            
            # Try backup file
            if self.backup_file.exists():
                try:
                    with open(self.backup_file, 'r') as f:
                        checkpoint = json.load(f)
                    
                    state = checkpoint.get('discovery_state', {})
                    stats = state.get('stats', {})
                    self.log.checkpoint(f"✓ Backup checkpoint loaded from {self.backup_file}")
                    self.log.checkpoint(f"  Generation: {stats.get('total_generated', 0)}, Timestamp: {time.ctime(checkpoint.get('timestamp', 0))}")
                    return state
                except Exception as backup_e:
                    self.log.error(f"✗ Failed to load backup checkpoint: {backup_e}")
                    
        return None
    
    def clear_checkpoint(self) -> None:
        """Remove checkpoint files after successful completion."""
        try:
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
                self.log.checkpoint(f"Cleared checkpoint: {self.checkpoint_file}")
            if self.backup_file.exists():
                self.backup_file.unlink()
                self.log.checkpoint(f"Cleared backup: {self.backup_file}")
        except Exception as e:
            self.log.error(f"Failed to clear checkpoints: {e}")
            raise Exception(f"Failed to clear checkpoints: {e}")
