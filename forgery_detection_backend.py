from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image, ExifTags
from PIL.ExifTags import TAGS
import os
import hashlib
import json
from datetime import datetime
import magic
import fitz  # PyMuPDF for PDF analysis
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class ImageForgeryDetector:
    """Advanced image forgery detection using multiple techniques"""
    
    def __init__(self):
        self.results = {}
    
    def analyze_image(self, image_data):
        """Main analysis function combining multiple detection methods"""
        try:
            # Convert image data to OpenCV format
            img_array = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            # Convert to PIL for EXIF analysis
            pil_img = Image.open(BytesIO(image_data))
            
            results = {
                'jpeg_analysis': self.analyze_jpeg_compression(img),
                'ela_analysis': self.error_level_analysis(img),
                'pixel_analysis': self.analyze_pixel_patterns(img),
                'metadata_analysis': self.analyze_metadata(pil_img),
                'noise_analysis': self.analyze_noise_patterns(img),
                'edge_analysis': self.analyze_edge_consistency(img),
                'overall_score': 0
            }
            
            # Calculate overall forgery probability
            results['overall_score'] = self.calculate_overall_score(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            return {'error': str(e)}
    
    def analyze_jpeg_compression(self, img):
        """Analyze JPEG compression artifacts for inconsistencies"""
        try:
            # Convert to grayscale for DCT analysis
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Analyze 8x8 blocks for compression artifacts
            h, w = gray.shape
            artifacts = []
            
            for y in range(0, h-8, 8):
                for x in range(0, w-8, 8):
                    block = gray[y:y+8, x:x+8].astype(np.float32)
                    
                    # Simple DCT coefficient analysis
                    dct = cv2.dct(block)
                    
                    # Check for unusual coefficient patterns
                    high_freq = np.sum(np.abs(dct[4:, 4:]))
                    low_freq = np.sum(np.abs(dct[:4, :4]))
                    
                    if low_freq > 0:
                        ratio = high_freq / low_freq
                        artifacts.append(ratio)
            
            if artifacts:
                mean_ratio = np.mean(artifacts)
                std_ratio = np.std(artifacts)
                
                # Higher variance suggests potential manipulation
                confidence = max(0, min(100, 100 - (std_ratio * 10)))
                
                return {
                    'confidence': round(confidence, 2),
                    'artifacts_detected': len([a for a in artifacts if a > mean_ratio + 2*std_ratio]),
                    'analysis': 'JPEG compression analysis completed'
                }
            else:
                return {
                    'confidence': 50.0,
                    'artifacts_detected': 0,
                    'analysis': 'Unable to analyze compression artifacts'
                }
                
        except Exception as e:
            return {
                'confidence': 0.0,
                'error': str(e),
                'analysis': 'Error in JPEG analysis'
            }
    
    def error_level_analysis(self, img):
        """Perform Error Level Analysis (ELA) to detect manipulated regions"""
        try:
            # Save image as JPEG with specific quality
            temp_path = 'temp_ela.jpg'
            cv2.imwrite(temp_path, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            # Reload the compressed image
            compressed = cv2.imread(temp_path)
            
            # Calculate absolute difference
            diff = cv2.absdiff(img, compressed)
            
            # Convert to grayscale and enhance
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            enhanced = cv2.equalizeHist(gray_diff)
            
            # Analyze error levels
            mean_error = np.mean(enhanced)
            std_error = np.std(enhanced)
            
            # Find regions with high error levels
            threshold = mean_error + 2 * std_error
            suspicious_regions = np.sum(enhanced > threshold)
            total_pixels = enhanced.shape[0] * enhanced.shape[1]
            
            suspicious_percentage = (suspicious_regions / total_pixels) * 100
            
            # Calculate confidence (lower suspicious percentage = higher confidence)
            confidence = max(0, min(100, 100 - suspicious_percentage * 2))
            
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return {
                'confidence': round(confidence, 2),
                'suspicious_regions': suspicious_percentage,
                'analysis': f'ELA analysis completed - {suspicious_percentage:.1f}% suspicious regions'
            }
            
        except Exception as e:
            return {
                'confidence': 0.0,
                'error': str(e),
                'analysis': 'Error in ELA analysis'
            }
    
    def analyze_pixel_patterns(self, img):
        """Analyze pixel patterns for cloning or copy-move forgeries"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Use ORB detector to find keypoints
            orb = cv2.ORB_create(nfeatures=1000)
            keypoints, descriptors = orb.detectAndCompute(gray, None)
            
            if descriptors is None or len(descriptors) < 10:
                return {
                    'confidence': 50.0,
                    'duplicated_features': 0,
                    'analysis': 'Insufficient features for analysis'
                }
            
            # Use FLANN matcher to find similar features
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH,
                              table_number=6,
                              key_size=12,
                              multi_probe_level=1)
            search_params = dict(checks=50)
            
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(descriptors, descriptors, k=3)
            
            # Filter matches to find potential duplications
            duplicates = 0
            for match_group in matches:
                if len(match_group) >= 2:
                    # Check if the second match is also very similar (potential duplication)
                    if match_group[1].distance < 50:  # Threshold for similarity
                        duplicates += 1
            
            duplication_ratio = duplicates / len(descriptors) if len(descriptors) > 0 else 0
            confidence = max(0, min(100, 100 - duplication_ratio * 200))
            
            return {
                'confidence': round(confidence, 2),
                'duplicated_features': duplicates,
                'analysis': f'Found {duplicates} potentially duplicated features'
            }
            
        except Exception as e:
            return {
                'confidence': 0.0,
                'error': str(e),
                'analysis': 'Error in pixel pattern analysis'
            }
    
    def analyze_metadata(self, pil_img):
        """Analyze image metadata for tampering signs"""
        try:
            exif_data = {}
            
            if hasattr(pil_img, '_getexif') and pil_img._getexif() is not None:
                exif = pil_img._getexif()
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    exif_data[tag] = value
            
            # Check for common manipulation signs
            suspicious_flags = 0
            total_checks = 0
            
            # Check for software manipulation indicators
            total_checks += 1
            if 'Software' in exif_data:
                software = str(exif_data['Software']).lower()
                editing_software = ['photoshop', 'gimp', 'paint.net', 'canva', 'pixlr']
                if any(editor in software for editor in editing_software):
                    suspicious_flags += 1
            
            # Check timestamp consistency
            total_checks += 1
            if 'DateTime' in exif_data and 'DateTimeOriginal' in exif_data:
                if exif_data['DateTime'] != exif_data['DateTimeOriginal']:
                    suspicious_flags += 1
            
            # Check for missing standard EXIF data
            total_checks += 1
            expected_tags = ['Make', 'Model', 'DateTime']
            missing_tags = sum(1 for tag in expected_tags if tag not in exif_data)
            if missing_tags > 1:
                suspicious_flags += 1
            
            confidence = max(0, min(100, 100 - (suspicious_flags / total_checks * 100)))
            
            return {
                'confidence': round(confidence, 2),
                'suspicious_flags': suspicious_flags,
                'total_tags': len(exif_data),
                'analysis': f'Metadata analysis: {suspicious_flags}/{total_checks} suspicious indicators'
            }
            
        except Exception as e:
            return {
                'confidence': 50.0,
                'error': str(e),
                'analysis': 'Error in metadata analysis'
            }
    
    def analyze_noise_patterns(self, img):
        """Analyze noise patterns for inconsistencies"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur and subtract to get noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            noise = cv2.absdiff(gray, blurred)
            
            # Divide image into blocks and analyze noise variance
            h, w = noise.shape
            block_size = 32
            variances = []
            
            for y in range(0, h-block_size, block_size):
                for x in range(0, w-block_size, block_size):
                    block = noise[y:y+block_size, x:x+block_size]
                    variance = np.var(block)
                    variances.append(variance)
            
            if variances:
                var_std = np.std(variances)
                var_mean = np.mean(variances)
                
                # Higher variance in noise variance suggests manipulation
                if var_mean > 0:
                    coefficient_of_variation = var_std / var_mean
                    confidence = max(0, min(100, 100 - coefficient_of_variation * 50))
                else:
                    confidence = 50.0
            else:
                confidence = 50.0
            
            return {
                'confidence': round(confidence, 2),
                'noise_variance': round(var_std, 2) if variances else 0,
                'analysis': 'Noise pattern analysis completed'
            }
            
        except Exception as e:
            return {
                'confidence': 0.0,
                'error': str(e),
                'analysis': 'Error in noise analysis'
            }
    
    def analyze_edge_consistency(self, img):
        """Analyze edge consistency across the image"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply Canny edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Divide into regions and analyze edge density
            h, w = edges.shape
            region_size = 64
            edge_densities = []
            
            for y in range(0, h-region_size, region_size):
                for x in range(0, w-region_size, region_size):
                    region = edges[y:y+region_size, x:x+region_size]
                    density = np.sum(region > 0) / (region_size * region_size)
                    edge_densities.append(density)
            
            if edge_densities:
                density_std = np.std(edge_densities)
                density_mean = np.mean(edge_densities)
                
                # More consistent edge density = higher confidence
                if density_mean > 0:
                    consistency = 1 - (density_std / density_mean)
                    confidence = max(0, min(100, consistency * 100))
                else:
                    confidence = 50.0
            else:
                confidence = 50.0
            
            return {
                'confidence': round(confidence, 2),
                'edge_consistency': round(consistency, 2) if edge_densities else 0,
                'analysis': 'Edge consistency analysis completed'
            }
            
        except Exception as e:
            return {
                'confidence': 0.0,
                'error': str(e),
                'analysis': 'Error in edge analysis'
            }
    
    def calculate_overall_score(self, results):
        """Calculate overall forgery detection score"""
        scores = []
        weights = {
            'jpeg_analysis': 0.2,
            'ela_analysis': 0.25,
            'pixel_analysis': 0.2,
            'metadata_analysis': 0.15,
            'noise_analysis': 0.1,
            'edge_analysis': 0.1
        }
        
        for analysis, weight in weights.items():
            if analysis in results and 'confidence' in results[analysis]:
                scores.append(results[analysis]['confidence'] * weight)
        
        return round(sum(scores), 2) if scores else 0.0


class DocumentForgeryDetector:
    """Document authenticity verification system"""
    
    def __init__(self):
        self.results = {}
    
    def analyze_document(self, file_data, filename):
        """Main document analysis function"""
        try:
            file_type = self.detect_file_type(file_data)
            
            results = {
                'file_info': self.analyze_file_info(file_data, filename, file_type),
                'content_analysis': self.analyze_content(file_data, file_type),
                'metadata_analysis': self.analyze_document_metadata(file_data, file_type),
                'structure_analysis': self.analyze_document_structure(file_data, file_type),
                'overall_score': 0
            }
            
            # Calculate overall authenticity score
            results['overall_score'] = self.calculate_overall_score(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing document: {str(e)}")
            return {'error': str(e)}
    
    def detect_file_type(self, file_data):
        """Detect file type using magic numbers"""
        try:
            mime = magic.from_buffer(file_data, mime=True)
            return mime
        except:
            return "application/octet-stream"
    
    def analyze_file_info(self, file_data, filename, file_type):
        """Analyze basic file information"""
        try:
            file_hash = hashlib.sha256(file_data).hexdigest()
            file_size = len(file_data)
            
            # Check file extension consistency
            extension_consistent = True
            if filename.lower().endswith('.pdf') and 'pdf' not in file_type:
                extension_consistent = False
            elif filename.lower().endswith(('.doc', '.docx')) and 'word' not in file_type:
                extension_consistent = False
            
            confidence = 90 if extension_consistent else 30
            
            return {
                'confidence': confidence,
                'file_type': file_type,
                'file_size': file_size,
                'file_hash': file_hash[:16] + '...',
                'extension_consistent': extension_consistent,
                'analysis': f'File type: {file_type}, Size: {file_size} bytes'
            }
            
        except Exception as e:
            return {
                'confidence': 0.0,
                'error': str(e),
                'analysis': 'Error in file info analysis'
            }
    
    def analyze_content(self, file_data, file_type):
        """Analyze document content for suspicious patterns"""
        try:
            if 'pdf' in file_type:
                return self.analyze_pdf_content(file_data)
            elif 'text' in file_type:
                return self.analyze_text_content(file_data)
            else:
                return {
                    'confidence': 50.0,
                    'analysis': f'Content analysis not supported for {file_type}'
                }
                
        except Exception as e:
            return {
                'confidence': 0.0,
                'error': str(e),
                'analysis': 'Error in content analysis'
            }
    
    def analyze_pdf_content(self, file_data):
        """Analyze PDF content for tampering signs"""
        try:
            doc = fitz.open(stream=file_data, filetype="pdf")
            
            suspicious_indicators = 0
            total_checks = 0
            
            # Check for text layer consistency
            total_checks += 1
            text_pages = 0
            for page in doc:
                if page.get_text().strip():
                    text_pages += 1
            
            if text_pages < len(doc) * 0.5:  # Less than 50% pages have text
                suspicious_indicators += 1
            
            # Check for embedded objects
            total_checks += 1
            embedded_objects = 0
            for page in doc:
                embedded_objects += len(page.get_images())
            
            if embedded_objects > len(doc) * 10:  # Unusually high number of objects
                suspicious_indicators += 1
            
            # Check creation/modification dates
            total_checks += 1
            metadata = doc.metadata
            if metadata.get('creationDate') and metadata.get('modDate'):
                # Suspicious if modified very recently after creation
                # This is a simplified check
                suspicious_indicators += 0  # Placeholder for actual date comparison
            
            confidence = max(0, min(100, 100 - (suspicious_indicators / total_checks * 100)))
            
            doc.close()
            
            return {
                'confidence': round(confidence, 2),
                'suspicious_indicators': suspicious_indicators,
                'total_pages': len(doc),
                'text_pages': text_pages,
                'embedded_objects': embedded_objects,
                'analysis': f'PDF analysis: {suspicious_indicators}/{total_checks} suspicious indicators'
            }
            
        except Exception as e:
            return {
                'confidence': 0.0,
                'error': str(e),
                'analysis': 'Error in PDF content analysis'
            }
    
    def analyze_text_content(self, file_data):
        """Analyze plain text content"""
        try:
            text = file_data.decode('utf-8', errors='ignore')
            
            # Basic text analysis
            word_count = len(text.split())
            char_count = len(text)
            line_count = len(text.split('\n'))
            
            # Check for unusual patterns
            suspicious_patterns = 0
            
            # Check for excessive whitespace manipulation
            if text.count('  ') > word_count * 0.1:
                suspicious_patterns += 1
            
            # Check for unusual character frequency
            printable_chars = sum(1 for c in text if c.isprintable())
            if char_count > 0 and printable_chars / char_count < 0.95:
                suspicious_patterns += 1
            
            confidence = max(0, min(100, 100 - suspicious_patterns * 30))
            
            return {
                'confidence': round(confidence, 2),
                'word_count': word_count,
                'char_count': char_count,
                'line_count': line_count,
                'suspicious_patterns': suspicious_patterns,
                'analysis': f'Text analysis: {suspicious_patterns} suspicious patterns detected'
            }
            
        except Exception as e:
            return {
                'confidence': 0.0,
                'error': str(e),
                'analysis': 'Error in text content analysis'
            }
    
    def analyze_document_metadata(self, file_data, file_type):
        """Analyze document metadata for authenticity"""
        try:
            if 'pdf' in file_type:
                doc = fitz.open(stream=file_data, filetype="pdf")
                metadata = doc.metadata
                doc.close()
                
                # Check metadata completeness and consistency
                expected_fields = ['title', 'author', 'creator', 'producer']
                present_fields = sum(1 for field in expected_fields if metadata.get(field))
                
                # Check for suspicious creation tools
                suspicious_tools = ['forge', 'fake', 'tamper', 'modify']
                creator_suspicious = any(tool in str(metadata.get('creator', '')).lower() 
                                       for tool in suspicious_tools)
                
                confidence = 80 if present_fields >= 2 and not creator_suspicious else 40
                
                return {
                    'confidence': confidence,
                    'present_fields': present_fields,
                    'total_expected': len(expected_fields),
                    'creator_suspicious': creator_suspicious,
                    'analysis': f'Metadata completeness: {present_fields}/{len(expected_fields)}'
                }
            else:
                return {
                    'confidence': 50.0,
                    'analysis': f'Metadata analysis not supported for {file_type}'
                }
                
        except Exception as e:
            return {
                'confidence': 0.0,
                'error': str(e),
                'analysis': 'Error in metadata analysis'
            }
    
    def analyze_document_structure(self, file_data, file_type):
        """Analyze document structure integrity"""
        try:
            if 'pdf' in file_type:
                # Check PDF structure
                try:
                    doc = fitz.open(stream=file_data, filetype="pdf")
                    page_count = len(doc)
                    
                    # Check for structural inconsistencies
                    inconsistencies = 0
                    
                    # Check page size consistency
                    page_sizes = []
                    for page in doc:
                        page_sizes.append((page.rect.width, page.rect.height))
                    
                    unique_sizes = len(set(page_sizes))
                    if unique_sizes > max(1, page_count * 0.3):  # Too many different sizes
                        inconsistencies += 1
                    
                    doc.close()
                    
                    confidence = max(0, min(100, 100 - inconsistencies * 50))
                    
                    return {
                        'confidence': round(confidence, 2),
                        'page_count': page_count,
                        'unique_page_sizes': unique_sizes,
                        'inconsistencies': inconsistencies,
                        'analysis': f'Structure analysis: {inconsistencies} inconsistencies found'
                    }
                    
                except:
                    return {
                        'confidence': 20.0,
                        'analysis': 'PDF structure appears corrupted or non-standard'
                    }
            else:
                return {
                    'confidence': 50.0,
                    'analysis': f'Structure analysis not supported for {file_type}'
                }
                
        except Exception as e:
            return {
                'confidence': 0.0,
                'error': str(e),
                'analysis': 'Error in structure analysis'
            }
    
    def calculate_overall_score(self, results):
        """Calculate overall document authenticity score"""
        scores = []
        weights = {
            'file_info': 0.2,
            'content_analysis': 0.3,
            'metadata_analysis': 0.25,
            'structure_analysis': 0.25
        }
        
        for analysis, weight in weights.items():
            if analysis in results and 'confidence' in results[analysis]:
                scores.append(results[analysis]['confidence'] * weight)
        
        return round(sum(scores), 2) if scores else 0.0


# Initialize detectors
image_detector = ImageForgeryDetector()
document_detector = DocumentForgeryDetector()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/api/analyze/image', methods=['POST'])
def analyze_image():
    """Analyze uploaded image for forgery"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read file data
        file_data = file.read()
        
        # Analyze image
        results = image_detector.analyze_image(file_data)
        
        return jsonify({
            'success': True,
            'filename': file.filename,
            'analysis_timestamp': datetime.now().isoformat(),
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Error in image analysis endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze/document', methods=['POST'])
def analyze_document():
    """Analyze uploaded document for authenticity"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read file data
        file_data = file.read()
        
        # Analyze document
        results = document_detector.analyze_document(file_data, file.filename)
        
        return jsonify({
            'success': True,
            'filename': file.filename,
            'analysis_timestamp': datetime.now().isoformat(),
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Error in document analysis endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze/batch', methods=['POST'])
def batch_analyze():
    """Analyze multiple files in batch"""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files uploaded'}), 400
        
        files = request.files.getlist('files')
        results = []
        
        for file in files:
            if file.filename == '':
                continue
                
            file_data = file.read()
            file_extension = file.filename.lower().split('.')[-1]
            
            if file_extension in ['jpg', 'jpeg', 'png', 'tiff', 'bmp']:
                analysis = image_detector.analyze_image(file_data)
                analysis_type = 'image'
            elif file_extension in ['pdf', 'doc', 'docx', 'txt']:
                analysis = document_detector.analyze_document(file_data, file.filename)
                analysis_type = 'document'
            else:
                analysis = {'error': 'Unsupported file type'}
                analysis_type = 'unknown'
            
            results.append({
                'filename': file.filename,
                'type': analysis_type,
                'analysis': analysis
            })
        
        return jsonify({
            'success': True,
            'batch_timestamp': datetime.now().isoformat(),
            'total_files': len(results),
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Error in batch analysis endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/report', methods=['POST'])
def generate_report():
    """Generate detailed analysis report"""
    try:
        data = request.get_json()
        
        if not data or 'results' not in data:
            return jsonify({'error': 'No analysis results provided'}), 400
        
        # Generate comprehensive report
        report = {
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'total_files_analyzed': len(data['results']),
                'suspicious_files': 0,
                'average_confidence': 0
            },
            'detailed_findings': [],
            'recommendations': []
        }
        
        total_confidence = 0
        for result in data['results']:
            if 'analysis' in result and 'overall_score' in result['analysis']:
                confidence = result['analysis']['overall_score']
                total_confidence += confidence
                
                if confidence < 70:
                    report['summary']['suspicious_files'] += 1
                    
                    finding = {
                        'filename': result['filename'],
                        'confidence': confidence,
                        'risk_level': 'HIGH' if confidence < 50 else 'MEDIUM',
                        'key_issues': []
                    }
                    
                    # Extract key issues from analysis
                    for analysis_type, analysis_data in result['analysis'].items():
                        if isinstance(analysis_data, dict) and 'confidence' in analysis_data:
                            if analysis_data['confidence'] < 60:
                                finding['key_issues'].append({
                                    'type': analysis_type,
                                    'confidence': analysis_data['confidence'],
                                    'description': analysis_data.get('analysis', 'No description')
                                })
                    
                    report['detailed_findings'].append(finding)
        
        if len(data['results']) > 0:
            report['summary']['average_confidence'] = round(total_confidence / len(data['results']), 2)
        
        # Generate recommendations
        if report['summary']['suspicious_files'] > 0:
            report['recommendations'].extend([
                'Review flagged files manually for verification',
                'Consider requesting original source files for comparison',
                'Implement additional verification procedures for low-confidence files'
            ])
        else:
            report['recommendations'].append('All analyzed files appear authentic based on digital forensics analysis')
        
        return jsonify({
            'success': True,
            'report': report
        })
        
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)