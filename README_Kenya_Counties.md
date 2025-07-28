# Kenya Counties Shapefile Extractor

This project provides tools to download Kenya county shapefiles and extract specific counties (Nakuru and Nyandarua) with their subcounties.

## Files Included

1. **`kenya_counties_extractor.py`** - Standalone Python script for extraction
2. **`kenya_counties_notebook.ipynb`** - Jupyter notebook with visualization capabilities
3. **`requirements.txt`** - Python dependencies
4. **`README_Kenya_Counties.md`** - This documentation

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Using the Python Script

Run the standalone script:
```bash
python kenya_counties_extractor.py
```

This will:
- Download Kenya county shapefiles from GADM
- Extract Nakuru and Nyandarua counties
- Save results in the `Ken_bounds` directory

### Option 2: Using the Jupyter Notebook

1. Start Jupyter:
```bash
jupyter notebook
```

2. Open `kenya_counties_notebook.ipynb`

3. Run all cells to:
- Download and extract the counties
- Visualize the results
- Generate maps and summaries

## Output Files

The extraction process creates the following files in the `Ken_bounds` directory:

- **`target_counties.shp`** - Shapefile with Nakuru and Nyandarua counties
- **`target_counties.geojson`** - GeoJSON format
- **`target_counties_summary.csv`** - Tabular data without geometry
- **`target_counties_map.png`** - Visualization map (from notebook)

## Features

- **Automatic Download**: Downloads Kenya county shapefiles from reliable sources
- **County Extraction**: Filters for specific counties (Nakuru and Nyandarua)
- **Subcounty Support**: Includes subcounty information when available
- **Multiple Formats**: Saves results in Shapefile, GeoJSON, and CSV formats
- **Visualization**: Creates maps and summaries (notebook version)
- **Error Handling**: Robust error handling and logging

## Target Counties

- **Nakuru County**: Located in the Rift Valley region
- **Nyandarua County**: Located in the Central region

Both counties include their respective subcounties and administrative boundaries.

## Data Sources

The script uses GADM (Global Administrative Areas) data, which provides:
- High-quality administrative boundaries
- Consistent naming conventions
- Multiple administrative levels (counties, subcounties, etc.)

## Troubleshooting

If you encounter issues:

1. **Download Failures**: The script tries multiple sources. Check your internet connection.
2. **Missing Counties**: The script will show available counties if target counties aren't found.
3. **Dependency Issues**: Ensure all requirements are installed correctly.

## Customization

To extract different counties, modify the `target_counties` list in the script:

```python
target_counties = ['Your County 1', 'Your County 2']
```

## License

This project is for educational and research purposes. Please respect the data source licenses. 