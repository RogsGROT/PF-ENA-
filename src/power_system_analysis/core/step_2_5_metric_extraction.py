import pandas as pd
import openpyxl
import xlrd
import os

def extract_metrics_from_excel(input_file, output_file):
    """
    Extract specific metrics from all sheets in an Excel file using direct cell references
    and organize them into a new Excel file by sheet name.
    
    Parameters:
    input_file (str): Path to the input Excel file
    output_file (str): Path to save the output Excel file
    """
    print(f"Reading input file: {input_file}")
    
    try:
        # Determine file format and use appropriate library
        if input_file.endswith('.xls'):
            # Use xlrd for .xls files
            workbook = xlrd.open_workbook(input_file)
            sheet_names = workbook.sheet_names()
            
            # Skip Sheet1 if it exists
            if 'Sheet1' in sheet_names:
                sheet_names.remove('Sheet1')
            
            print(f"Found {len(sheet_names)} sheets to process")
            
            # Process each sheet
            results = []
            for sheet_name in sheet_names:
                print(f"Processing sheet: {sheet_name}")
                sheet = workbook.sheet_by_name(sheet_name)
                
                # Create a dictionary to store the metrics for this sheet
                sheet_metrics = {'Sheet Name': sheet_name}
                
                # Define the metrics and their cell references (using 0-based indexing for xlrd)
                # These match exactly what's in the Excel file
                metrics_row1 = [
                    ('Cyclicity', (26, 0)),  # A27
                    ('Linkage Density', (26, 1)),  # B27
                    ('Predator/Prey Ratio', (26, 2)),  # C27
                    ('Generalization', (26, 3)),  # D27
                    ('Vulnerability', (26, 4)),  # E27
                    ('Actors', (26, 5)),  # F27
                    ('Links', (26, 6)),  # G27
                    ('Number of Predators', (26, 7)),  # H27
                    ('Number of Prey', (26, 8)),  # I27
                    ('Connectance', (26, 9)),  # J27
                    ('Number of Special Prey', (26, 10)),  # K27
                    ('Fraction of Special Predators', (26, 11))  # L27
                ]
                
                metrics_row2 = [
                    ('Finn Cycling Index', (28, 0)),  # A29
                    ('Mean Path Length', (28, 1)),  # B29
                    ('Average Mutual Information', (28, 2)),  # C29
                    ('Ascendency', (28, 3)),  # D29
                    ('Developmental Capacity', (28, 4)),  # E29
                    ('Total System Overhead', (28, 5)),  # F29
                    ('Total System Through Flow', (28, 6)),  # G29
                    ('Alpha', (28, 7)),  # H29
                    ('Robustness', (28, 8)),  # I29
                    ('Shannon Index', (28, 9))  # J29
                ]
                
                # Extract first row metrics
                for metric_name, (row, col) in metrics_row1:
                    try:
                        if row < sheet.nrows and col < sheet.ncols:
                            value = sheet.cell_value(row, col)
                            sheet_metrics[metric_name] = value
                        else:
                            print(f"Warning: Cell ({row},{col}) is out of bounds for sheet {sheet_name}")
                            sheet_metrics[metric_name] = None
                    except Exception as e:
                        print(f"Error extracting {metric_name} from sheet {sheet_name}: {e}")
                        sheet_metrics[metric_name] = None
                
                # Extract second row metrics
                for metric_name, (row, col) in metrics_row2:
                    try:
                        if row < sheet.nrows and col < sheet.ncols:
                            value = sheet.cell_value(row, col)
                            sheet_metrics[metric_name] = value
                        else:
                            print(f"Warning: Cell ({row},{col}) is out of bounds for sheet {sheet_name}")
                            sheet_metrics[metric_name] = None
                    except Exception as e:
                        print(f"Error extracting {metric_name} from sheet {sheet_name}: {e}")
                        sheet_metrics[metric_name] = None
                
                results.append(sheet_metrics)
            
        else:
            # Use openpyxl for .xlsx files
            workbook = openpyxl.load_workbook(input_file, data_only=True)
            sheet_names = workbook.sheetnames
            
            # Skip Sheet1 if it exists
            if 'Sheet1' in sheet_names:
                sheet_names.remove('Sheet1')
            
            print(f"Found {len(sheet_names)} sheets to process")
            
            # Define the metrics and their cell references
            # These match exactly what's in the Excel file
            metrics_row1 = [
                ('Cyclicity', 'A27'),
                ('Linkage Density', 'B27'),
                ('Predator/Prey Ratio', 'C27'),
                ('Generalization', 'D27'),
                ('Vulnerability', 'E27'),
                ('Actors', 'F27'),
                ('Links', 'G27'),
                ('Number of Predators', 'H27'),
                ('Number of Prey', 'I27'),
                ('Connectance', 'J27'),
                ('Number of Special Prey', 'K27'),
                ('Fraction of Special Predators', 'L27')
            ]
            
            metrics_row2 = [
                ('Finn Cycling Index', 'A29'),
                ('Mean Path Length', 'B29'),
                ('Average Mutual Information', 'C29'),
                ('Ascendency', 'D29'),
                ('Developmental Capacity', 'E29'),
                ('Total System Overhead', 'F29'),
                ('Total System Through Flow', 'G29'),
                ('Alpha', 'H29'),
                ('Robustness', 'I29'),
                ('Shannon Index', 'J29')
            ]
            
            # Process each sheet
            results = []
            for sheet_name in sheet_names:
                print(f"Processing sheet: {sheet_name}")
                sheet = workbook[sheet_name]
                
                # Create a dictionary to store the metrics for this sheet
                sheet_metrics = {'Sheet Name': sheet_name}
                
                # Extract first row metrics
                for metric_name, cell_ref in metrics_row1:
                    try:
                        value = sheet[cell_ref].value
                        sheet_metrics[metric_name] = value
                    except Exception as e:
                        print(f"Error extracting {metric_name} from sheet {sheet_name}: {e}")
                        sheet_metrics[metric_name] = None
                
                # Extract second row metrics
                for metric_name, cell_ref in metrics_row2:
                    try:
                        value = sheet[cell_ref].value
                        sheet_metrics[metric_name] = value
                    except Exception as e:
                        print(f"Error extracting {metric_name} from sheet {sheet_name}: {e}")
                        sheet_metrics[metric_name] = None
                
                results.append(sheet_metrics)
        
        # Convert results to DataFrame
        df_results = pd.DataFrame(results)
        
        # Define the column order to ensure all metrics appear in the correct order
        column_order = ['Sheet Name']
        # Add all row1 metrics
        column_order.extend([metric[0] for metric in metrics_row1])
        # Add all row2 metrics
        column_order.extend([metric[0] for metric in metrics_row2])
        
        # Reorder the DataFrame columns
        for col in column_order:
            if col not in df_results.columns:
                print(f"Warning: Column {col} not found in results")
        
        # Only use columns that exist in the DataFrame
        valid_columns = [col for col in column_order if col in df_results.columns]
        df_results = df_results[valid_columns]
        
        # Save the results to a new Excel file
        print(f"Saving results to {output_file}")
        df_results.to_excel(output_file, index=False)
        
        print(f"Successfully extracted metrics from {len(results)} sheets and saved to {output_file}")
        return True
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return False


if __name__ == "__main__":
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try to get the project root directory (3 levels up from this script)
    try:
        project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
    except Exception:
        # Fallback to current directory
        project_root = script_dir
    
    # Define input and output file paths
    input_file = os.path.join(project_root, "Analyzed_output_flow_matrices.xls")
    output_file = os.path.join(project_root, "Extracted_Metrics_by_Sheet.xlsx")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found!")
        
        # Try looking in the current directory as a fallback
        input_file = "Analyzed_output_flow_matrices.xls"
        output_file = "Extracted_Metrics_by_Sheet.xlsx"
        
        if not os.path.exists(input_file):
            print(f"Input file also not found in current directory!")
            exit(1)
    
    # Extract metrics
    success = extract_metrics_from_excel(input_file, output_file)
    
    if success:
        print("Script completed successfully.")
    else:
        print("Script failed to complete.")