import ollama
import re
import base64
from PIL import Image
import io
import json
import os
import sys

class FiberLengthDetector:
    def __init__(self, model_name='llava-phi3'):
        """
        Initialize the Fiber Length Detector with Ollama model
        """
        self.model_name = model_name
        try:
            self.client = ollama.Client()
            print(f"Connected to Ollama with model: {model_name}")
        except Exception as e:
            print(f"Error connecting to Ollama: {e}")
            raise Exception(f"Cannot connect to Ollama: {e}")
    
    
    def process_image(self, image_path):
        """
        Process a single image to extract fiber length
        
        Args:
            image_path: Path to the image file
            
        Returns:
            dict: Analysis results
        """
        try:
            # Read and convert image to bytes
            image_bytes = self._image_to_bytes(image_path)
            
            # Extract number using Ollama model
            result = self._extract_number_from_image_bytes(image_bytes, image_path)
            
            return result
            
        except Exception as e:
            return {
                'detected_length': 'Not detected',
                'unit': 'N/A',
                'confidence': 0,
                'method': 'Ollama Model',
                'raw_text': f'Error: {str(e)}',
                'additional_numbers': [],
                'error': str(e)
            }
    
    def process_two_images(self, image_path1, image_path2):
        """
        Process two images and calculate the difference (like your Colab code)
        
        Args:
            image_path1: Path to first image
            image_path2: Path to second image
            
        Returns:
            dict: Analysis results with difference calculation
        """
        try:
            print(f"Processing two images for comparison...")
            print(f"Image 1: {os.path.basename(image_path1)}")
            print(f"Image 2: {os.path.basename(image_path2)}")
            
            # Process both images (like your Colab)
            image1_bytes = self._image_to_bytes(image_path1)
            image2_bytes = self._image_to_bytes(image_path2)
            
            # Extract numbers from both images
            num1 = self._extract_number_from_image_bytes(image1_bytes, image_path1)
            num2 = self._extract_number_from_image_bytes(image2_bytes, image_path2)
            
            print(f"Raw model output for {os.path.basename(image_path1)}:")
            print(num1.get('raw_text', 'No response'))
            
            print(f"Raw model output for {os.path.basename(image_path2)}:")
            print(num2.get('raw_text', 'No response'))
            
            # Calculate difference (like your Colab)
            length1 = num1.get('detected_length')
            length2 = num2.get('detected_length')
            
            difference = None
            if (length1 and length1 != 'Not detected' and length1 is not None and 
                length2 and length2 != 'Not detected' and length2 is not None):
                try:
                    val1 = float(length1)
                    val2 = float(length2)
                    difference = abs(val1 - val2)
                    print(f"Fiber length difference: {difference} meters")
                except (ValueError, TypeError):
                    print("Could not calculate difference due to invalid numbers")
                    difference = None
            else:
                print("Could not calculate difference due to missing number(s)")
            
            return {
                'image1_result': num1,
                'image2_result': num2,
                'image1_path': image_path1,
                'image2_path': image_path2,
                'difference': difference,
                'difference_unit': 'meters' if difference is not None else 'N/A',
                'method': 'Dual Image Analysis'
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'image1_result': None,
                'image2_result': None,
                'difference': None,
                'method': 'Dual Image Analysis - Failed'
            }
    
    def _image_to_bytes(self, image_path):
        """
        Convert image file to bytes for Ollama processing
        """
        try:
            with open(image_path, 'rb') as image_file:
                image_bytes = image_file.read()
            return image_bytes
        except Exception as e:
            raise Exception(f"Failed to read image file: {str(e)}")
    
    def _extract_number_from_image_bytes(self, image_bytes, image_name='uploaded_image'):
        """
        Extract handwritten number from image using Ollama model (matching your Colab function)
        """
        try:
            # Send chat request to Ollama model (same as your Colab)
            response = self.client.chat(
                model=self.model_name,
                messages=[{
                    'role': 'user',
                    'content': 'Extract the handwritten number in meters from this image.',
                    'images': [image_bytes]
                }]
            )
            
            # Extract the content of the model's response
            content = response['message']['content']
            raw_text = content.strip()
            
            print(f"Raw model output for {os.path.basename(image_name) if hasattr(image_name, '__len__') else image_name}:")
            print(content)
            
            # Use regular expression to find a numerical value (same as your Colab)
            # optionally followed by "m" or "meters" (case-insensitive)
            match = re.search(r'(\d+(?:\.\d+)?)(?:\s*m| meters)?', content.lower())
            
            detected_length = None
            confidence = 50  # Base confidence
            additional_numbers = []
            
            if match:
                # If a match is found, convert the captured number to a float and return it
                detected_length = float(match.group(1))
                
                # Find all numbers for additional_numbers
                all_matches = re.findall(r'(\d+(?:\.\d+)?)', content.lower())
                additional_numbers = [float(m) for m in all_matches[1:]]  # Skip the first one
                
                # Calculate confidence based on clarity
                confidence = self._calculate_confidence(content, detected_length)
            else:
                # If no match is found, print a message and return None
                print(f"No number found in {os.path.basename(image_name) if hasattr(image_name, '__len__') else image_name}")
            
            return {
                'detected_length': detected_length,
                'unit': 'meters' if detected_length is not None else 'N/A',
                'confidence': confidence,
                'method': 'Ollama Model',
                'raw_text': raw_text,
                'additional_numbers': additional_numbers,
                'model_used': self.model_name
            }
            
        except Exception as e:
            raise Exception(f"Failed to process image with Ollama: {str(e)}")
    
    def _calculate_confidence(self, model_output, detected_value):
        """
        Calculate confidence score based on model output clarity
        """
        confidence = 50  # Base confidence
        
        # Increase confidence if the model mentions specific measurement terms
        measurement_terms = ['meter', 'meters', 'm', 'measurement', 'length', 'fiber']
        for term in measurement_terms:
            if term.lower() in model_output.lower():
                confidence += 10
                break
        
        # Increase confidence if the number appears with units
        if str(detected_value) in model_output and ('m' in model_output.lower() or 'meter' in model_output.lower()):
            confidence += 20
        
        # Increase confidence if model seems certain
        certainty_words = ['clearly', 'shows', 'reads', 'indicates', 'visible']
        for word in certainty_words:
            if word.lower() in model_output.lower():
                confidence += 5
        
        # Decrease confidence if model expresses uncertainty
        uncertainty_words = ['might', 'appears', 'seems', 'possibly', 'unclear']
        for word in uncertainty_words:
            if word.lower() in model_output.lower():
                confidence -= 10
        
        # Ensure confidence is within valid range
        confidence = max(0, min(100, confidence))
        
        return confidence