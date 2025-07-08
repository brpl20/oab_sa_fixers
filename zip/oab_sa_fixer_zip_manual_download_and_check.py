#!/usr/bin/env python3
"""
Download Problem Images Script
Downloads images from S3 based on error logs from ZIP Code Extraction
"""

import os
import json
import asyncio
import aiohttp
import boto3
from pathlib import Path
from datetime import datetime
import logging
from botocore.exceptions import ClientError, NoCredentialsError
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProblemImageDownloader:
    def __init__(self, base_path="/Volumes/BPSSD/codessd/oab_sa_fixers"):
        self.base_path = Path(base_path)
        self.results_dir = self.base_path / "zip_results"
        self.problem_images_dir = self.base_path / "problem_images"
        
        # Create problem images directory
        self.problem_images_dir.mkdir(exist_ok=True)
        
        # Load AWS credentials
        self.load_aws_credentials()
        
        # Initialize S3 client
        self.s3_client = None
        self.init_s3_client()
        
        # Statistics
        self.total_errors = 0
        self.downloaded_count = 0
        self.failed_downloads = 0
        
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
            
    def find_error_files(self) -> List[Path]:
        """Find all error JSON files in the results directory"""
        error_files = []
        
        # Look for error files
        for error_file in self.results_dir.glob("*error*.json"):
            error_files.append(error_file)
            
        logger.info(f"📋 Found {len(error_files)} error files")
        return error_files
        
    def load_error_data(self, error_files: List[Path]) -> List[Dict[str, Any]]:
        """Load all error data from JSON files"""
        all_errors = []
        
        for error_file in error_files:
            try:
                logger.info(f"📖 Loading errors from: {error_file}")
                with open(error_file, 'r', encoding='utf-8') as f:
                    errors = json.load(f)
                    
                if isinstance(errors, list):
                    all_errors.extend(errors)
                else:
                    all_errors.append(errors)
                    
                logger.info(f"  ➡️ Loaded {len(errors)} errors")
                
            except Exception as e:
                logger.error(f"❌ Error loading {error_file}: {e}")
                
        logger.info(f"📊 Total errors loaded: {len(all_errors)}")
        return all_errors
        
    def extract_unique_images(self, error_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract unique images from error data"""
        unique_images = {}
        
        for error in error_data:
            image_name = error.get('image')
            lawyer_id = error.get('id')
            
            if image_name and lawyer_id:
                if image_name not in unique_images:
                    unique_images[image_name] = {
                        'image': image_name,
                        'lawyer_id': lawyer_id,
                        'error_type': error.get('error', 'Unknown error'),
                        'timestamp': error.get('timestamp')
                    }
                    
        logger.info(f"🖼️  Found {len(unique_images)} unique problem images")
        return list(unique_images.values())
        
    def download_image_from_s3(self, image_key: str, local_path: Path, lawyer_id: str) -> bool:
        """Download image from S3 bucket"""
        try:
            logger.info(f"📥 Downloading lawyer {lawyer_id}: {image_key}")
            
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
            
    def create_info_file(self, image_info: Dict[str, Any], image_path: Path):
        """Create a text file with information about the problem image"""
        info_file = image_path.parent / f"{image_path.stem}_info.txt"
        
        try:
            with open(info_file, 'w', encoding='utf-8') as f:
                f.write(f"Problem Image Information\n")
                f.write(f"========================\n\n")
                f.write(f"Lawyer ID: {image_info['lawyer_id']}\n")
                f.write(f"Image: {image_info['image']}\n")
                f.write(f"Error Type: {image_info['error_type']}\n")
                f.write(f"Timestamp: {image_info['timestamp']}\n")
                f.write(f"Downloaded: {datetime.now().isoformat()}\n\n")
                f.write(f"Notes:\n")
                f.write(f"- This image failed ZIP code extraction\n")
                f.write(f"- Manual inspection required\n")
                f.write(f"- Check address field visibility and quality\n")
                
        except Exception as e:
            logger.error(f"❌ Error creating info file: {e}")
            
    def download_problem_images(self, problem_images: List[Dict[str, Any]]):
        """Download all problem images"""
        logger.info(f"🚀 Starting download of {len(problem_images)} problem images")
        
        for i, image_info in enumerate(problem_images, 1):
            image_name = image_info['image']
            lawyer_id = image_info['lawyer_id']
            
            # Create filename with lawyer ID for easier identification
            file_extension = Path(image_name).suffix
            safe_filename = f"lawyer_{lawyer_id}_{image_name}"
            local_path = self.problem_images_dir / safe_filename
            
            logger.info(f"\n📸 [{i}/{len(problem_images)}] Processing: {image_name}")
            
            # Download image
            if self.download_image_from_s3(image_name, local_path, lawyer_id):
                self.downloaded_count += 1
                
                # Create info file
                self.create_info_file(image_info, local_path)
                
            else:
                self.failed_downloads += 1
                
        # Summary
        logger.info(f"\n🎉 DOWNLOAD COMPLETE!")
        logger.info(f"✅ Successfully downloaded: {self.downloaded_count}")
        logger.info(f"❌ Failed downloads: {self.failed_downloads}")
        logger.info(f"📊 Success rate: {self.downloaded_count/(self.downloaded_count+self.failed_downloads)*100:.1f}%" if (self.downloaded_count+self.failed_downloads) > 0 else "N/A")
        logger.info(f"📂 Images saved to: {self.problem_images_dir}")
        
    def create_summary_report(self, error_data: List[Dict[str, Any]]):
        """Create a summary report of all errors"""
        summary_file = self.problem_images_dir / "error_summary_report.txt"
        
        try:
            # Group errors by type
            error_types = {}
            for error in error_data:
                error_type = error.get('error', 'Unknown error')
                if error_type not in error_types:
                    error_types[error_type] = []
                error_types[error_type].append(error)
                
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"Error Summary Report\n")
                f.write(f"===================\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Errors: {len(error_data)}\n")
                f.write(f"Unique Images: {self.downloaded_count + self.failed_downloads}\n\n")
                
                f.write(f"Error Types:\n")
                f.write(f"------------\n")
                for error_type, errors in error_types.items():
                    f.write(f"{error_type}: {len(errors)} occurrences\n")
                    
                f.write(f"\nDetailed Errors:\n")
                f.write(f"----------------\n")
                for error_type, errors in error_types.items():
                    f.write(f"\n{error_type}:\n")
                    for error in errors[:5]:  # Show first 5 examples
                        f.write(f"  - Lawyer {error.get('id')}: {error.get('image')}\n")
                    if len(errors) > 5:
                        f.write(f"  ... and {len(errors) - 5} more\n")
                        
            logger.info(f"📋 Summary report created: {summary_file}")
            
        except Exception as e:
            logger.error(f"❌ Error creating summary report: {e}")
            
    def run(self):
        """Main execution method"""
        try:
            logger.info(f"🚀 Starting Problem Images Download")
            logger.info(f"📂 Results directory: {self.results_dir}")
            logger.info(f"💾 Download directory: {self.problem_images_dir}")
            
            # Find error files
            error_files = self.find_error_files()
            
            if not error_files:
                logger.error("❌ No error files found")
                return
                
            # Load error data
            error_data = self.load_error_data(error_files)
            
            if not error_data:
                logger.error("❌ No error data loaded")
                return
                
            # Extract unique images
            problem_images = self.extract_unique_images(error_data)
            
            if not problem_images:
                logger.error("❌ No problem images found")
                return
                
            # Download images
            self.download_problem_images(problem_images)
            
            # Create summary report
            self.create_summary_report(error_data)
            
        except Exception as e:
            logger.error(f"❌ Critical error in main execution: {e}")
            raise

def main():
    """Main entry point"""
    try:
        downloader = ProblemImageDownloader()
        downloader.run()
    except KeyboardInterrupt:
        logger.info("\n⏹️  Process interrupted by user")
    except Exception as e:
        logger.error(f"💥 Critical error: {e}")

if __name__ == "__main__":
    # Check dependencies
    try:
        import boto3
        logger.info("✅ All dependencies available")
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        logger.error("Install with: pip install boto3")
        exit(1)
    
    # Run the main function
    main()