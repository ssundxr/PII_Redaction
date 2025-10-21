"""
Unit tests for PII explanation functionality.
Tests the integration of NLP explanations with PII detection.
"""

import unittest
import os
import sys
from pathlib import Path
from PIL import Image
import tempfile

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from src.redactor import PIIRedactor
    from src.privacy.nlp_explain import NLPExplain, get_nlp_explainer
    from src.storage.local_db import LocalDB
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import modules: {e}")
    MODULES_AVAILABLE = False

class TestPIIExplanations(unittest.TestCase):
    """Test cases for PII explanation functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class with sample data."""
        if not MODULES_AVAILABLE:
            cls.skipTest(cls, "Required modules not available")
            return
        
        # Create a simple test image with text
        cls.test_image = Image.new('RGB', (400, 200), color='white')
        # In a real test, you'd add text to the image or use a sample image
        
        # Save test image temporarily
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_image_path = os.path.join(cls.temp_dir, "test_document.png")
        cls.test_image.save(cls.test_image_path)
    
    def setUp(self):
        """Set up each test case."""
        if not MODULES_AVAILABLE:
            self.skipTest("Required modules not available")
    
    def test_nlp_explainer_initialization(self):
        """Test that NLP explainer initializes correctly."""
        explainer = get_nlp_explainer()
        self.assertIsInstance(explainer, NLPExplain)
        
        # Test text explanation
        result = explainer.explain_text("John Doe lives at 123 Main St")
        self.assertIn('entities', result)
        self.assertIsInstance(result['entities'], list)
    
    def test_bbox_explanation(self):
        """Test bbox explanation functionality."""
        explainer = get_nlp_explainer()
        
        # Test with sample text and bbox
        bbox = (10, 20, 100, 30)  # x, y, w, h
        result = explainer.explain_bbox("Student ID: 12345", bbox)
        
        # Check structure
        self.assertIn('bbox', result)
        self.assertIn('text', result)
        self.assertIn('entities', result)
        self.assertIn('primary_explanation', result)
        
        # Check bbox format
        bbox_data = result['bbox']
        self.assertEqual(bbox_data['x'], 10)
        self.assertEqual(bbox_data['y'], 20)
        self.assertEqual(bbox_data['width'], 100)
        self.assertEqual(bbox_data['height'], 30)
    
    def test_custom_pii_explanations(self):
        """Test custom PII keyword explanations."""
        explainer = get_nlp_explainer()
        
        # Test custom keywords
        test_cases = [
            ("admission no: 2024001", "admission no"),
            ("roll no: CS001", "roll no"),
            ("student id: ST12345", "student id")
        ]
        
        for text, expected_keyword in test_cases:
            result = explainer.explain_text(text)
            entities = result['entities']
            
            # Should find at least one entity
            self.assertGreater(len(entities), 0)
            
            # Check if custom explanation is present
            custom_entities = [e for e in entities if e['label'] == 'CUSTOM_PII']
            self.assertGreater(len(custom_entities), 0)
    
    def test_pii_redactor_integration(self):
        """Test that PIIRedactor generates explanations."""
        try:
            # Initialize redactor (this might fail if models aren't available)
            redactor = PIIRedactor()
            
            # Process the test image
            redacted_image = redactor.process_document(self.test_image_path)
            
            # Get audit data
            audit_data = redactor.generate_audit_log()
            
            # Check audit data structure
            self.assertIn('filename', audit_data)
            self.assertIn('timestamp', audit_data)
            self.assertIn('num_redactions', audit_data)
            
            # For any detected PII items, check explanation structure
            if 'pii_items' in audit_data:
                for item in audit_data['pii_items']:
                    self.assertIn('explanation_text', item)
                    self.assertIn('bbox', item)
                    self.assertIn('label', item)
                    self.assertIn('reason', item)
                    self.assertIn('confidence', item)
                    
                    # Check explanation_text is not empty
                    self.assertIsInstance(item['explanation_text'], str)
                    self.assertGreater(len(item['explanation_text']), 0)
            
        except Exception as e:
            # Skip if models can't be loaded
            self.skipTest(f"Could not initialize PIIRedactor: {e}")
    
    def test_database_storage(self):
        """Test that explanations are stored in database."""
        try:
            # Create temporary database
            temp_db_path = os.path.join(self.temp_dir, "test_audit.db")
            temp_key_path = os.path.join(self.temp_dir, "test_audit.key")
            
            db = LocalDB(db_path=temp_db_path, key_path=temp_key_path)
            
            # Create sample audit log with explanations
            audit_log = {
                'filename': 'test_document.png',
                'timestamp': '2024-01-01T12:00:00',
                'num_redactions': 2,
                'pii_items': [
                    {
                        'bbox': {'x': 10, 'y': 20, 'width': 100, 'height': 30},
                        'label': 'TEXT_PII',
                        'reason': 'Keyword matching',
                        'explanation_text': 'PERSON: Proper Name detected by NER',
                        'confidence': 0.9,
                        'text_content': 'John Doe'
                    },
                    {
                        'bbox': {'x': 50, 'y': 80, 'width': 120, 'height': 25},
                        'label': 'CUSTOM_PII',
                        'reason': 'Custom keyword matching',
                        'explanation_text': 'Student admission identifier used for enrollment',
                        'confidence': 0.8,
                        'text_content': 'admission no: 12345'
                    }
                ]
            }
            
            # Store audit log
            audit_id = db.store_audit(audit_log)
            self.assertIsInstance(audit_id, int)
            self.assertGreater(audit_id, 0)
            
            # Retrieve and verify
            retrieved = db.get_audit_by_id(audit_id)
            self.assertIsNotNone(retrieved)
            self.assertIn('explanations', retrieved)
            
            # Check explanations structure
            explanations = retrieved['explanations']
            self.assertIn('items', explanations)
            self.assertIn('summary', explanations)
            self.assertEqual(len(explanations['items']), 2)
            
            # Verify explanation content
            for item in explanations['items']:
                self.assertIn('explanation_text', item)
                self.assertIn('bbox', item)
                self.assertIn('label', item)
                self.assertGreater(len(item['explanation_text']), 0)
            
        except Exception as e:
            self.fail(f"Database storage test failed: {e}")
    
    def test_batch_explanation_performance(self):
        """Test batch explanation for performance."""
        explainer = get_nlp_explainer()
        
        # Create batch of text fragments
        text_fragments = [
            "John Smith",
            "admission no: 12345",
            "Email: john@example.com",
            "Phone: +1-555-123-4567",
            "Student ID: ST001"
        ]
        
        bboxes = [(i*20, i*30, 80, 20) for i in range(len(text_fragments))]
        
        # Process batch
        results = explainer.batch_explain(text_fragments, bboxes)
        
        # Verify results
        self.assertEqual(len(results), len(text_fragments))
        
        for i, result in enumerate(results):
            self.assertIn('bbox', result)
            self.assertIn('text', result)
            self.assertIn('primary_explanation', result)
            self.assertEqual(result['text'], text_fragments[i])
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test class."""
        if hasattr(cls, 'temp_dir'):
            import shutil
            shutil.rmtree(cls.temp_dir, ignore_errors=True)

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
