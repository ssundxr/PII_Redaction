"""
Local Database with Encryption for PII Audit Logs
Stores encrypted audit trails with explanations for detected PII.
"""

import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from cryptography.fernet import Fernet
import os

logger = logging.getLogger(__name__)

class LocalDB:
    """Encrypted local database for storing PII detection audit logs."""
    
    def __init__(self, db_path: str = "audit_logs.db", key_path: str = "audit.key"):
        """Initialize the local database with encryption."""
        self.db_path = Path(db_path)
        self.key_path = Path(key_path)
        self.cipher_suite = self._get_or_create_cipher()
        self._init_database()
    
    def _get_or_create_cipher(self) -> Fernet:
        """Get or create encryption key."""
        if self.key_path.exists():
            with open(self.key_path, 'rb') as f:
                key = f.read()
        else:
            key = Fernet.generate_key()
            with open(self.key_path, 'wb') as f:
                f.write(key)
            os.chmod(self.key_path, 0o600)  # Restrict key file permissions
        return Fernet(key)
    
    def _init_database(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create audit_logs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    num_redactions INTEGER NOT NULL,
                    encrypted_data BLOB NOT NULL,
                    explanations TEXT DEFAULT NULL
                )
            ''')
            
            # Check if explanations column exists, add if missing
            cursor.execute("PRAGMA table_info(audit_logs)")
            columns = [col[1] for col in cursor.fetchall()]
            if 'explanations' not in columns:
                cursor.execute('ALTER TABLE audit_logs ADD COLUMN explanations TEXT')
                logger.info("Added explanations column to audit_logs table")
            
            conn.commit()
    
    def _encrypt_data(self, data: Dict[str, Any]) -> bytes:
        """Encrypt sensitive audit data."""
        json_data = json.dumps(data, ensure_ascii=False)
        return self.cipher_suite.encrypt(json_data.encode('utf-8'))
    
    def _decrypt_data(self, encrypted_data: bytes) -> Dict[str, Any]:
        """Decrypt audit data."""
        decrypted_json = self.cipher_suite.decrypt(encrypted_data)
        return json.loads(decrypted_json.decode('utf-8'))
    
    def store_audit(self, audit_log: Dict[str, Any]) -> int:
        """
        Store encrypted audit log with explanations.
        
        Args:
            audit_log: Dictionary containing audit information
                      {filename, timestamp, num_redactions, pii_items: [...]}
        
        Returns:
            int: The ID of the stored audit log
        """
        try:
            # Extract explanations for separate storage
            pii_items = audit_log.get('pii_items', [])
            explanations = {
                'items': [
                    {
                        'bbox': item.get('bbox', {}),
                        'label': item.get('label', ''),
                        'reason': item.get('reason', ''),
                        'explanation_text': item.get('explanation_text', ''),
                        'confidence': item.get('confidence', 0.0)
                    }
                    for item in pii_items
                ],
                'summary': {
                    'total_items': len(pii_items),
                    'types_detected': list(set(item.get('label', '') for item in pii_items))
                }
            }
            
            # Encrypt sensitive data (original audit log)
            encrypted_data = self._encrypt_data(audit_log)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO audit_logs 
                    (filename, timestamp, num_redactions, encrypted_data, explanations)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    audit_log.get('filename', 'unknown'),
                    audit_log.get('timestamp', datetime.now().isoformat()),
                    audit_log.get('num_redactions', 0),
                    encrypted_data,
                    json.dumps(explanations)
                ))
                
                audit_id = cursor.lastrowid
                conn.commit()
                
                logger.info(f"Stored audit log with ID {audit_id}")
                return audit_id
                
        except Exception as e:
            logger.error(f"Failed to store audit log: {e}")
            raise
    
    def get_audit_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve decrypted audit history with explanations.
        
        Args:
            limit: Maximum number of records to retrieve
            
        Returns:
            List of audit log dictionaries with explanations
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id, filename, timestamp, num_redactions, 
                           encrypted_data, explanations
                    FROM audit_logs 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (limit,))
                
                results = []
                for row in cursor.fetchall():
                    audit_id, filename, timestamp, num_redactions, encrypted_data, explanations_json = row
                    
                    # Decrypt main data
                    decrypted_data = self._decrypt_data(encrypted_data)
                    
                    # Parse explanations
                    explanations = {}
                    if explanations_json:
                        try:
                            explanations = json.loads(explanations_json)
                        except json.JSONDecodeError:
                            logger.warning(f"Could not parse explanations for audit {audit_id}")
                    
                    # Combine data
                    result = {
                        'id': audit_id,
                        'filename': filename,
                        'timestamp': timestamp,
                        'num_redactions': num_redactions,
                        'explanations': explanations,
                        **decrypted_data
                    }
                    
                    results.append(result)
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to retrieve audit history: {e}")
            raise
    
    def get_audit_by_id(self, audit_id: int) -> Optional[Dict[str, Any]]:
        """Get specific audit log by ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT filename, timestamp, num_redactions, 
                           encrypted_data, explanations
                    FROM audit_logs 
                    WHERE id = ?
                ''', (audit_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                filename, timestamp, num_redactions, encrypted_data, explanations_json = row
                
                # Decrypt and parse data
                decrypted_data = self._decrypt_data(encrypted_data)
                explanations = {}
                if explanations_json:
                    try:
                        explanations = json.loads(explanations_json)
                    except json.JSONDecodeError:
                        logger.warning(f"Could not parse explanations for audit {audit_id}")
                
                return {
                    'id': audit_id,
                    'filename': filename,
                    'timestamp': timestamp,
                    'num_redactions': num_redactions,
                    'explanations': explanations,
                    **decrypted_data
                }
                
        except Exception as e:
            logger.error(f"Failed to retrieve audit {audit_id}: {e}")
            return None
    
    def delete_old_audits(self, days_old: int = 90):
        """Delete audit logs older than specified days."""
        try:
            cutoff_date = datetime.now().timestamp() - (days_old * 24 * 3600)
            cutoff_iso = datetime.fromtimestamp(cutoff_date).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    DELETE FROM audit_logs 
                    WHERE timestamp < ?
                ''', (cutoff_iso,))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                logger.info(f"Deleted {deleted_count} old audit logs")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Failed to delete old audits: {e}")
            return 0

# Global instance
_db_instance = None

def get_local_db() -> LocalDB:
    """Get or create global database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = LocalDB()
    return _db_instance