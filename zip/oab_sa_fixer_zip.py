#!/usr/bin/env python3
"""
ZIP Code Extractor for OAB Lawyer Images
Downloads images from S3, extracts ZIP codes using OCR, and validates them using BrasilAPI
"""

import os
import json
import time
import re
import asyncio
import aiohttp
import boto3
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
import pytesseract
from botocore.exceptions import ClientError, NoCredentialsError
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ZipCodeExtractor:
    def __init__(self, base_path="/Volumes/BPSSD/codessd/oab_sa_fixers"):
        self.base_path = Path(base_path)
        self.images_dir = self.base_path / "temp_images"
        self.results_dir = self.base_path / "zip_results"
        
        # Create directories
        self.images_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        # Load environment variables
        self.load_aws_credentials()
        
        # Initialize S3 client
        self.s3_client = None
        self.init_s3_client()
        
        # Results storage
        self.processed_lawyers = []
        self.error_logs = []
        
    def load_aws_credentials(self):
        """Load AWS credentials from environment or .env file"""
        try:
            # Try to load from .env file
            env_file = self.base_path / ".env"
            if env_file.exists():
                with open(env_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and '=' in line and not line.startswith('#'):
                            key, value = line.split('=', 1)
                            os.environ[key.strip()] = value.strip().strip('"').strip("'")
            
            # Get credentials
            self.aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
            self.aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
            self.aws_bucket = os.getenv('AWS_BUCKET')
            self.aws_region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
            
            if not all([self.aws_access_key, self.aws_secret_key, self.aws_bucket]):
                raise ValueError("Missing required AWS credentials")
                
            logger.info(f"✅ AWS credentials loaded - Bucket: {self.aws_bucket}")
            
        except Exception as e:
            logger.error(f"❌ Error loading AWS credentials: {e}")
            raise
            
    def init_s3_client(self):
        """Initialize S3 client with credentials"""
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key,
                region_name=self.aws_region
            )
            
            # Test connection
            self.s3_client.head_bucket(Bucket=self.aws_bucket)
            logger.info(f"✅ S3 client initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing S3 client: {e}")
            raise
            
    def download_image_from_s3(self, image_key, local_path):
        """Download image from S3 bucket"""
        try:
            logger.info(f"📥 Downloading: {image_key}")
            
            self.s3_client.download_file(
                self.aws_bucket, 
                image_key, 
                str(local_path)
            )
            
            # Verify file was downloaded and is valid
            if local_path.exists() and local_path.stat().st_size > 0:
                logger.info(f"✅ Downloaded: {local_path}")
                return True
            else:
                logger.error(f"❌ Download failed or empty file: {image_key}")
                return False
                
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                logger.error(f"❌ Image not found in S3: {image_key}")
            else:
                logger.error(f"❌ S3 error downloading {image_key}: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Error downloading {image_key}: {e}")
            return False
            
    def crop_address_field(self, image_path):
        """Crop the address field from the OAB image using the geometry from original code"""
        try:
            # Read image
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError("Could not read image")
                
            # Address field geometry from original code: "490x82+0+157"
            # Format: widthxheight+x_offset+y_offset
            x, y, w, h = 0, 157, 490, 82
            
            # Crop the address field
            cropped = img[y:y+h, x:x+w]
            
            # Save cropped image
            crop_path = image_path.parent / f"{image_path.stem}_address_crop.jpg"
            cv2.imwrite(str(crop_path), cropped)
            
            logger.info(f"✂️  Address field cropped: {crop_path}")
            return crop_path
            
        except Exception as e:
            logger.error(f"❌ Error cropping address field: {e}")
            return None
            
    def enhance_image_for_ocr(self, image_path):
        """Enhance image for better OCR results focusing on ZIP code extraction"""
        try:
            # Read image
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError("Could not read image")
                
            # Convert to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Increase resolution for better OCR (400% scale for ZIP codes)
            scale_percent = 400
            width = int(img_rgb.shape[1] * scale_percent / 100)
            height = int(img_rgb.shape[0] * scale_percent / 100)
            img_upscaled = cv2.resize(img_rgb, (width, height), interpolation=cv2.INTER_CUBIC)
            
            # Apply bilateral filter to reduce noise while preserving edges
            img_bilateral = cv2.bilateralFilter(img_upscaled, 9, 75, 75)
            
            # Convert to grayscale and apply CLAHE
            gray = cv2.cvtColor(img_bilateral, cv2.COLOR_RGB2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_enhanced = clahe.apply(gray)
            
            # Save enhanced image
            enhanced_path = image_path.parent / f"{image_path.stem}_enhanced.jpg"
            cv2.imwrite(str(enhanced_path), img_enhanced)
            
            return enhanced_path
            
        except Exception as e:
            logger.error(f"❌ Error enhancing image: {e}")
            return image_path  # Return original if enhancement fails
            
    def extract_zip_code_with_ocr(self, image_path):
        """Extract ZIP code from image using OCR with specific configuration for Brazilian CEP"""
        try:
            # Enhance image first
            enhanced_path = self.enhance_image_for_ocr(image_path)
            
            # OCR configuration optimized for ZIP codes (Brazilian CEP format: 12345-678)
            custom_config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789- '
            
            # Perform OCR
            text = pytesseract.image_to_string(
                str(enhanced_path), 
                config=custom_config, 
                lang='por'
            )
            
            logger.info(f"🔍 Raw OCR text: {repr(text)}")
            
            # Extract ZIP codes using regex patterns
            zip_codes = self.extract_zip_patterns(text)
            
            if zip_codes:
                logger.info(f"📮 Found ZIP codes: {zip_codes}")
                return zip_codes
            else:
                logger.warning(f"❌ No ZIP code found in OCR text")
                return []
                
        except Exception as e:
            logger.error(f"❌ OCR error: {e}")
            return []
            
    def extract_zip_patterns(self, text):
        """Extract Brazilian ZIP code patterns from text"""
        zip_codes = []
        
        # Brazilian ZIP code patterns
        patterns = [
            r'\d{5}-\d{3}',          # 12345-678 (standard format)
            r'\d{8}',                # 12345678 (no hyphen)
            r'\d{5}\s*-\s*\d{3}',    # 12345 - 678 (with spaces)
            r'\d{2}\.\d{3}-\d{3}',   # 12.345-678 (with dot)
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                # Clean the match - remove spaces, dots, keep only digits and hyphen
                clean_zip = re.sub(r'[^\d-]', '', match)
                
                # Ensure it has exactly 8 digits
                digits_only = re.sub(r'[^\d]', '', clean_zip)
                if len(digits_only) == 8:
                    # Format as standard CEP (12345-678)
                    formatted_zip = f"{digits_only[:5]}-{digits_only[5:]}"
                    if formatted_zip not in zip_codes:
                        zip_codes.append(formatted_zip)
        
        return zip_codes
        
    async def validate_zip_code(self, zip_code, session):
        """Validate ZIP code using BrasilAPI"""
        try:
            # Remove hyphen for API call
            clean_zip = zip_code.replace('-', '')
            
            # BrasilAPI CEP endpoint
            url = f"https://brasilapi.com.br/api/cep/v1/{clean_zip}"
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Construct address from API response
                    address_parts = []
                    if data.get('street'):
                        address_parts.append(data['street'])
                    if data.get('neighborhood'):
                        address_parts.append(data['neighborhood'])
                    if data.get('city'):
                        address_parts.append(data['city'])
                    if data.get('state'):
                        address_parts.append(data['state'])
                        
                    address_from_zip = ', '.join(address_parts)
                    
                    logger.info(f"✅ Valid ZIP {zip_code}: {address_from_zip}")
                    
                    return {
                        'valid': True,
                        'zip_code': zip_code,
                        'address_from_zip': address_from_zip,
                        'api_response': data
                    }
                else:
                    logger.warning(f"❌ Invalid ZIP {zip_code}: HTTP {response.status}")
                    return {'valid': False, 'zip_code': zip_code, 'error': f"HTTP {response.status}"}
                    
        except Exception as e:
            logger.error(f"❌ Error validating ZIP {zip_code}: {e}")
            return {'valid': False, 'zip_code': zip_code, 'error': str(e)}
            
    async def process_single_lawyer(self, lawyer_data, session):
        """Process a single lawyer's image to extract ZIP code"""
        lawyer_id = lawyer_data['id']
        image_filename = lawyer_data['cna_picture']
        
        logger.info(f"\n👨‍💼 Processing lawyer ID {lawyer_id}: {image_filename}")
        
        try:
            # Download image from S3
            local_image_path = self.images_dir / image_filename
            download_success = self.download_image_from_s3(image_filename, local_image_path)
            
            if not download_success:
                error = {
                    'id': lawyer_id,
                    'image': image_filename,
                    'error': 'Failed to download image from S3',
                    'timestamp': datetime.now().isoformat()
                }
                self.error_logs.append(error)
                return None
                
            # Crop address field
            cropped_path = self.crop_address_field(local_image_path)
            if not cropped_path:
                error = {
                    'id': lawyer_id,
                    'image': image_filename,
                    'error': 'Failed to crop address field',
                    'timestamp': datetime.now().isoformat()
                }
                self.error_logs.append(error)
                return None
                
            # Extract ZIP codes using OCR
            zip_codes = self.extract_zip_code_with_ocr(cropped_path)
            
            if not zip_codes:
                error = {
                    'id': lawyer_id,
                    'image': image_filename,
                    'error': 'No ZIP code found in OCR',
                    'timestamp': datetime.now().isoformat()
                }
                self.error_logs.append(error)
                return None
                
            # Validate each found ZIP code
            for zip_code in zip_codes:
                validation_result = await self.validate_zip_code(zip_code, session)
                
                if validation_result['valid']:
                    # Success! Add to processed lawyers
                    result = {
                        'id': lawyer_id,
                        'cna_picture': image_filename,
                        'zip_code': validation_result['zip_code'],
                        'address_from_zip': validation_result['address_from_zip'],
                        'processed_at': datetime.now().isoformat(),
                        'api_response': validation_result['api_response']
                    }
                    
                    logger.info(f"✅ SUCCESS for lawyer {lawyer_id}: {validation_result['zip_code']}")
                    return result
                    
            # If we get here, no valid ZIP codes were found
            error = {
                'id': lawyer_id,
                'image': image_filename,
                'error': f'No valid ZIP codes found. Extracted: {zip_codes}',
                'timestamp': datetime.now().isoformat()
            }
            self.error_logs.append(error)
            return None
            
        except Exception as e:
            error = {
                'id': lawyer_id,
                'image': image_filename,
                'error': f'Unexpected error: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
            self.error_logs.append(error)
            logger.error(f"❌ Unexpected error processing lawyer {lawyer_id}: {e}")
            return None
            
        finally:
            # Cleanup temporary files
            try:
                if local_image_path.exists():
                    local_image_path.unlink()
                for temp_file in self.images_dir.glob(f"{local_image_path.stem}*"):
                    temp_file.unlink()
            except:
                pass
                
    async def process_lawyers_batch(self, lawyers_data, batch_size=5):
        """Process lawyers in batches to avoid overwhelming the API"""
        
        async with aiohttp.ClientSession() as session:
            for i in range(0, len(lawyers_data), batch_size):
                batch = lawyers_data[i:i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (len(lawyers_data) + batch_size - 1) // batch_size
                
                logger.info(f"\n📦 Processing batch {batch_num}/{total_batches} ({len(batch)} lawyers)")
                
                # Process batch concurrently
                tasks = [self.process_single_lawyer(lawyer, session) for lawyer in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Handle results
                for result in results:
                    if isinstance(result, Exception):
                        logger.error(f"❌ Batch task exception: {result}")
                    elif result is not None:
                        self.processed_lawyers.append(result)
                        
                # Add delay between batches to be respectful to the API
                if i + batch_size < len(lawyers_data):
                    logger.info(f"⏳ Waiting 2 seconds before next batch...")
                    await asyncio.sleep(2)
                    
                # Save progress periodically
                if batch_num % 10 == 0:
                    self.save_progress()
                    
    def save_progress(self):
        """Save current progress to files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save successful results
        if self.processed_lawyers:
            results_file = self.results_dir / f"zip_extraction_results_{timestamp}.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(self.processed_lawyers, f, ensure_ascii=False, indent=2)
            logger.info(f"💾 Saved {len(self.processed_lawyers)} results to: {results_file}")
            
        # Save errors
        if self.error_logs:
            errors_file = self.results_dir / f"zip_extraction_errors_{timestamp}.json"
            with open(errors_file, 'w', encoding='utf-8') as f:
                json.dump(self.error_logs, f, ensure_ascii=False, indent=2)
            logger.info(f"💾 Saved {len(self.error_logs)} errors to: {errors_file}")
            
    def load_lawyers_data(self, json_file_path):
        """Load lawyers data from JSON file"""
        try:
            json_path = Path(json_file_path)
            if not json_path.exists():
                raise FileNotFoundError(f"File not found: {json_file_path}")
                
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            logger.info(f"📋 Loaded {len(data)} lawyers from: {json_path}")
            return data
            
        except Exception as e:
            logger.error(f"❌ Error loading lawyers data: {e}")
            raise
            
    async def run(self, json_file_path=None):
        """Main execution method"""
        try:
            # Default path if not provided
            if json_file_path is None:
                json_file_path = self.base_path / "zip" / "invalid_cep_lawyers_20250702_1121.json"
                
            logger.info(f"🚀 Starting ZIP Code Extraction")
            logger.info(f"📂 Input file: {json_file_path}")
            logger.info(f"🪣 S3 Bucket: {self.aws_bucket}")
            logger.info(f"💾 Results dir: {self.results_dir}")
            
            # Load lawyers data
            lawyers_data = self.load_lawyers_data(json_file_path)
            
            if not lawyers_data:
                logger.error("❌ No lawyers data to process")
                return
                
            # Process all lawyers
            await self.process_lawyers_batch(lawyers_data, batch_size=5)
            
            # Final save
            self.save_progress()
            
            # Summary
            logger.info(f"\n🎉 EXTRACTION COMPLETE!")
            logger.info(f"✅ Successfully processed: {len(self.processed_lawyers)}")
            logger.info(f"❌ Errors: {len(self.error_logs)}")
            logger.info(f"📊 Success rate: {len(self.processed_lawyers)/(len(self.processed_lawyers)+len(self.error_logs))*100:.1f}%" if (len(self.processed_lawyers)+len(self.error_logs)) > 0 else "N/A")
            
        except Exception as e:
            logger.error(f"❌ Critical error in main execution: {e}")
            raise

async def main():
    """Main entry point"""
    try:
        extractor = ZipCodeExtractor()
        await extractor.run()
    except KeyboardInterrupt:
        logger.info("\n⏹️  Process interrupted by user")
    except Exception as e:
        logger.error(f"💥 Critical error: {e}")

if __name__ == "__main__":
    # Check dependencies
    try:
        import cv2
        import pytesseract
        import boto3
        import aiohttp
        logger.info("✅ All dependencies available")
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        logger.error("Install with: pip install opencv-python pytesseract boto3 aiohttp pillow")
        exit(1)
    
    # Run the async main function
    asyncio.run(main())