#!/usr/bin/env python3
"""
Kenya Counties Shapefile Extractor
Downloads Kenya county shapefiles and extracts specific counties with their subcounties.
"""

import os
import requests
import zipfile
import geopandas as gpd
import pandas as pd
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KenyaCountiesExtractor:
    def __init__(self, output_dir="Ken_bounds"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Kenya county shapefile sources
        self.shapefile_urls = [
            "https://data.humdata.org/dataset/kenya-county-boundaries/resource/6d78c4e7-4d4f-4e4f-8f4f-4f4f4f4f4f4f/download/kenya-counties.zip",
            "https://geoportal.rcmrd.org/downloads/kenya-counties.zip"
        ]
        
        # Alternative: Use GADM data
        self.gadm_url = "https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_KEN_shp.zip"
        
    def download_shapefiles(self, use_gadm=True):
        """Download Kenya county shapefiles from GADM or other sources."""
        try:
            if use_gadm:
                logger.info("Downloading Kenya shapefiles from GADM...")
                response = requests.get(self.gadm_url, stream=True)
                response.raise_for_status()
                
                zip_path = self.output_dir / "kenya_gadm.zip"
                with open(zip_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Extract the zip file
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.output_dir)
                
                # Find the county level shapefile (admin level 1)
                shapefile_path = None
                for file in self.output_dir.glob("*.shp"):
                    if "KEN_1" in file.name:
                        shapefile_path = file
                        break
                
                if shapefile_path:
                    logger.info(f"Found shapefile: {shapefile_path}")
                    return shapefile_path
                else:
                    logger.error("Could not find county level shapefile in GADM data")
                    return None
                    
            else:
                # Try alternative sources
                for url in self.shapefile_urls:
                    try:
                        logger.info(f"Trying to download from: {url}")
                        response = requests.get(url, stream=True)
                        response.raise_for_status()
                        
                        zip_path = self.output_dir / "kenya_counties.zip"
                        with open(zip_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        
                        # Extract and return
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            zip_ref.extractall(self.output_dir)
                        
                        # Find shapefile
                        shapefile_path = next(self.output_dir.glob("*.shp"), None)
                        if shapefile_path:
                            return shapefile_path
                            
                    except Exception as e:
                        logger.warning(f"Failed to download from {url}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error downloading shapefiles: {e}")
            return None
    
    def load_counties_data(self, shapefile_path):
        """Load counties data from shapefile."""
        try:
            gdf = gpd.read_file(shapefile_path)
            logger.info(f"Loaded {len(gdf)} counties")
            logger.info(f"Columns: {list(gdf.columns)}")
            return gdf
        except Exception as e:
            logger.error(f"Error loading shapefile: {e}")
            return None
    
    def extract_target_counties(self, gdf, target_counties=['Nakuru', 'Nyandarua']):
        """Extract specific counties from the dataset."""
        try:
            # Try different possible column names for county names
            name_columns = ['NAME_1', 'NAME', 'county_name', 'COUNTY_NAME', 'name', 'County']
            
            county_col = None
            for col in name_columns:
                if col in gdf.columns:
                    county_col = col
                    break
            
            if county_col is None:
                logger.error(f"Could not find county name column. Available columns: {list(gdf.columns)}")
                return None
            
            # Filter for target counties (case-insensitive)
            target_counties_lower = [county.lower() for county in target_counties]
            gdf['county_lower'] = gdf[county_col].str.lower()
            
            filtered_gdf = gdf[gdf['county_lower'].isin(target_counties_lower)].copy()
            
            if len(filtered_gdf) == 0:
                logger.warning("No target counties found. Available counties:")
                available_counties = gdf[county_col].unique()
                for county in available_counties:
                    logger.info(f"  - {county}")
                return None
            
            logger.info(f"Found {len(filtered_gdf)} target counties")
            for county in filtered_gdf[county_col].unique():
                logger.info(f"  - {county}")
            
            return filtered_gdf.drop(columns=['county_lower'])
            
        except Exception as e:
            logger.error(f"Error extracting target counties: {e}")
            return None
    
    def get_subcounties(self, gdf):
        """Get subcounties for the target counties."""
        try:
            # Try to find subcounty level data
            subcounty_columns = ['NAME_2', 'subcounty_name', 'SUBCOUNTY_NAME', 'subcounty']
            
            subcounty_col = None
            for col in subcounty_columns:
                if col in gdf.columns:
                    subcounty_col = col
                    break
            
            if subcounty_col:
                logger.info(f"Found subcounty column: {subcounty_col}")
                subcounties = gdf[subcounty_col].unique()
                logger.info(f"Subcounties found: {list(subcounties)}")
                return gdf
            else:
                logger.warning("No subcounty information found in the data")
                return gdf
                
        except Exception as e:
            logger.error(f"Error getting subcounties: {e}")
            return gdf
    
    def save_results(self, gdf, filename="target_counties"):
        """Save the extracted counties to shapefile and GeoJSON."""
        try:
            # Save as shapefile
            shapefile_path = self.output_dir / f"{filename}.shp"
            gdf.to_file(shapefile_path)
            logger.info(f"Saved shapefile: {shapefile_path}")
            
            # Save as GeoJSON
            geojson_path = self.output_dir / f"{filename}.geojson"
            gdf.to_file(geojson_path, driver='GeoJSON')
            logger.info(f"Saved GeoJSON: {geojson_path}")
            
            # Save summary as CSV
            csv_path = self.output_dir / f"{filename}_summary.csv"
            summary_df = gdf.drop(columns=['geometry']).copy()
            summary_df.to_csv(csv_path, index=False)
            logger.info(f"Saved summary CSV: {csv_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return False
    
    def run_extraction(self, target_counties=['Nakuru', 'Nyandarua']):
        """Run the complete extraction process."""
        logger.info("Starting Kenya counties extraction process...")
        
        # Step 1: Download shapefiles
        shapefile_path = self.download_shapefiles()
        if not shapefile_path:
            logger.error("Failed to download shapefiles")
            return False
        
        # Step 2: Load data
        gdf = self.load_counties_data(shapefile_path)
        if gdf is None:
            logger.error("Failed to load counties data")
            return False
        
        # Step 3: Extract target counties
        target_gdf = self.extract_target_counties(gdf, target_counties)
        if target_gdf is None:
            logger.error("Failed to extract target counties")
            return False
        
        # Step 4: Get subcounties
        final_gdf = self.get_subcounties(target_gdf)
        
        # Step 5: Save results
        success = self.save_results(final_gdf)
        
        if success:
            logger.info("Extraction completed successfully!")
            logger.info(f"Results saved in: {self.output_dir}")
        else:
            logger.error("Failed to save results")
        
        return success

def main():
    """Main function to run the extraction."""
    extractor = KenyaCountiesExtractor()
    
    # Define target counties
    target_counties = ['Nakuru', 'Nyandarua']
    
    # Run extraction
    success = extractor.run_extraction(target_counties)
    
    if success:
        print(f"\n✅ Extraction completed successfully!")
        print(f"📁 Results saved in: {extractor.output_dir}")
        print(f"🎯 Target counties: {', '.join(target_counties)}")
    else:
        print("\n❌ Extraction failed. Check logs for details.")

if __name__ == "__main__":
    main() 