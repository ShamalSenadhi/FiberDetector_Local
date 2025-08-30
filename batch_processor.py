import os
import glob
import json
import time
from datetime import datetime
from fiber_detector import FiberLengthDetector

class BatchFiberProcessor:
    def __init__(self, model_name="llava-phi3"):
        print("🚀 Initializing Batch Fiber Processor...")
        self.detector = FiberLengthDetector(model_name)
        
    def process_directory(self, input_dir, output_file="batch_results.json"):
        """Process all images in a directory"""
        print(f"\n📁 Scanning directory: {input_dir}")
        
        # Supported image formats
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.gif']
        image_files = []
        
        for ext in extensions:
            pattern = os.path.join(input_dir, ext)
            image_files.extend(glob.glob(pattern, recursive=False))
            # Also check uppercase
            pattern_upper = os.path.join(input_dir, ext.upper())
            image_files.extend(glob.glob(pattern_upper, recursive=False))
        
        if not image_files:
            print(f"❌ No image files found in {input_dir}")
            print(f"   Supported formats: {', '.join(extensions)}")
            return
        
        # Remove duplicates and sort
        image_files = sorted(list(set(image_files)))
        
        print(f"✅ Found {len(image_files)} image files")
        for i, img in enumerate(image_files, 1):
            print(f"   {i}. {os.path.basename(img)}")
        
        results = []
        total_files = len(image_files)
        start_time = time.time()
        
        print(f"\n🔄 Starting batch processing...")
        print("=" * 60)
        
        for i, image_path in enumerate(image_files, 1):
            print(f"\n[{i}/{total_files}] Processing: {os.path.basename(image_path)}")
            
            file_start_time = time.time()
            result = self.detector.process_image(image_path)
            file_end_time = time.time()
            
            processing_time = file_end_time - file_start_time
            
            if result:
                result['filename'] = os.path.basename(image_path)
                result['filepath'] = image_path
                result['processed_at'] = datetime.now().isoformat()
                result['processing_time_seconds'] = round(processing_time, 2)
                results.append(result)
                
                # Quick summary
                length = result.get('detected_length')
                unit = result.get('unit', '')
                confidence = result.get('confidence', 0)
                
                if length:
                    print(f"   ✅ Found: {length} {unit} (confidence: {confidence}%)")
                else:
                    print(f"   ❌ No measurement detected")
            else:
                print(f"   💥 Failed to process")
            
            print(f"   ⏱️  Processing time: {processing_time:.1f}s")
            
            # Estimate remaining time
            if i < total_files:
                avg_time = (time.time() - start_time) / i
                remaining_time = avg_time * (total_files - i)
                mins, secs = divmod(remaining_time, 60)
                print(f"   🕐 Estimated remaining: {int(mins)}m {int(secs)}s")
        
        # Save results to JSON file
        output_path = os.path.join(input_dir, output_file)
        
        summary = {
            "processing_summary": {
                "total_files": total_files,
                "successfully_processed": len(results),
                "failed_files": total_files - len(results),
                "total_processing_time_seconds": round(time.time() - start_time, 2),
                "average_time_per_file": round((time.time() - start_time) / total_files, 2),
                "processed_at": datetime.now().isoformat(),
                "input_directory": input_dir
            },
            "results": results
        }
        
        try:
            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"\n{'='*60}")
            print("📊 BATCH PROCESSING COMPLETE!")
            print(f"{'='*60}")
            print(f"✅ Successfully processed: {len(results)}/{total_files} files")
            print(f"⏱️  Total time: {(time.time() - start_time)/60:.1f} minutes")
            print(f"💾 Results saved to: {output_path}")
            
            # Show summary of detections
            detected_count = sum(1 for r in results if r.get('detected_length'))
            print(f"🔍 Measurements detected in: {detected_count}/{len(results)} files")
            
            if detected_count > 0:
                print(f"\n📏 Detected measurements:")
                for result in results:
                    if result.get('detected_length'):
                        length = result.get('detected_length')
                        unit = result.get('unit', '')
                        confidence = result.get('confidence', 0)
                        filename = result.get('filename', 'Unknown')
                        print(f"   • {filename}: {length} {unit} ({confidence}% confidence)")
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")

def main():
    print("🚀 Batch Fiber Length Processor")
    print("=" * 40)
    
    processor = BatchFiberProcessor()
    
    while True:
        print(f"\n📁 Enter directory path containing images:")
        print("   (or 'quit' to exit)")
        
        input_directory = input("➤ ").strip().strip('"').strip("'")
        
        if input_directory.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not input_directory:
            print("❌ Please enter a valid directory path")
            continue
        
        if not os.path.exists(input_directory):
            print(f"❌ Directory not found: {input_directory}")
            continue
        
        if not os.path.isdir(input_directory):
            print(f"❌ Path is not a directory: {input_directory}")
            continue
        
        # Ask for output filename
        print(f"\n💾 Enter output filename (default: batch_results.json):")
        output_file = input("➤ ").strip()
        if not output_file:
            output_file = "batch_results.json"
        
        # Ensure .json extension
        if not output_file.endswith('.json'):
            output_file += '.json'
        
        # Confirm before starting
        print(f"\n📋 Processing Summary:")
        print(f"   Input Directory: {input_directory}")
        print(f"   Output File: {output_file}")
        
        confirm = input("\nStart processing? (y/n): ").strip().lower()
        if confirm in ['y', 'yes']:
            processor.process_directory(input_directory, output_file)
        else:
            print("❌ Processing cancelled")

if __name__ == "__main__":
    main()