import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from price_predictor import HousePricePredictor
import json
import os

class ModernHousePriceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("House Price Predictor")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Load the model
        try:
            self.predictor = HousePricePredictor()
            self.feature_info = self.predictor.get_feature_requirements()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load the model: {e}")
            root.destroy()
            return
        
        self.setup_styles()
        self.create_widgets()
        
        # History of predictions
        self.prediction_history = []
    
    def setup_styles(self):
        """Set up custom styles for widgets"""
        style = ttk.Style()
        style.configure('Modern.TFrame', background='#f0f0f0')
        style.configure('Modern.TLabel', background='#f0f0f0', font=('Arial', 10))
        style.configure('Modern.TButton', font=('Arial', 10, 'bold'), padding=5)
        style.configure('Title.TLabel', background='#f0f0f0', font=('Arial', 16, 'bold'))
        style.configure('Result.TLabel', background='#f0f0f0', font=('Arial', 14))
    
    def create_widgets(self):
        """Create and arrange all GUI widgets"""
        # Main container
        main_frame = ttk.Frame(self.root, style='Modern.TFrame', padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Title
        title_label = ttk.Label(main_frame, text="House Price Predictor", 
                              style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Input frame (left side)
        input_frame = ttk.Frame(main_frame, style='Modern.TFrame', padding="10")
        input_frame.grid(row=1, column=0, sticky="nsew")
        
        # Create input fields
        self.input_vars = {}
        row = 0
        for feature, description in self.feature_info.items():
            label = ttk.Label(input_frame, text=f"{feature}:", style='Modern.TLabel')
            label.grid(row=row, column=0, sticky="w", pady=2)
            
            tooltip = description
            
            if feature in ['OverallQual', 'OverallCond']:
                var = tk.StringVar()
                spinbox = ttk.Spinbox(input_frame, from_=1, to=10, width=20,
                                    textvariable=var)
                spinbox.grid(row=row, column=1, sticky="w", pady=2)
                self.input_vars[feature] = var
            elif feature == 'Neighborhood':
                var = tk.StringVar()
                entry = ttk.Entry(input_frame, width=20, textvariable=var)
                entry.grid(row=row, column=1, sticky="w", pady=2)
                self.input_vars[feature] = var
            else:
                var = tk.StringVar()
                entry = ttk.Entry(input_frame, width=20, textvariable=var)
                entry.grid(row=row, column=1, sticky="w", pady=2)
                self.input_vars[feature] = var
            
            # Add tooltip
            self.create_tooltip(label, tooltip)
            row += 1
        
        # Buttons frame
        button_frame = ttk.Frame(input_frame, style='Modern.TFrame')
        button_frame.grid(row=row, column=0, columnspan=2, pady=20)
        
        predict_btn = ttk.Button(button_frame, text="Predict Price",
                               command=self.predict_single, style='Modern.TButton')
        predict_btn.grid(row=0, column=0, padx=5)
        
        clear_btn = ttk.Button(button_frame, text="Clear Fields",
                             command=self.clear_fields, style='Modern.TButton')
        clear_btn.grid(row=0, column=1, padx=5)
        
        load_btn = ttk.Button(button_frame, text="Load CSV",
                            command=self.load_csv, style='Modern.TButton')
        load_btn.grid(row=0, column=2, padx=5)
        
        # Result label
        self.result_label = ttk.Label(input_frame, text="", style='Result.TLabel')
        self.result_label.grid(row=row+1, column=0, columnspan=2, pady=10)
        
        # Visualization frame (right side)
        viz_frame = ttk.Frame(main_frame, style='Modern.TFrame', padding="10")
        viz_frame.grid(row=1, column=1, sticky="nsew")
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
    
    def create_tooltip(self, widget, text):
        """Create a tooltip for a widget"""
        def show_tooltip(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            
            label = ttk.Label(tooltip, text=text, background="#ffffe0", 
                            relief='solid', borderwidth=1)
            label.pack()
            
            def hide_tooltip():
                tooltip.destroy()
            
            tooltip.after(3000, hide_tooltip)
            widget.tooltip = tooltip
        
        def hide_tooltip(event):
            if hasattr(widget, 'tooltip'):
                widget.tooltip.destroy()
        
        widget.bind('<Enter>', show_tooltip)
        widget.bind('<Leave>', hide_tooltip)
    
    def predict_single(self):
        """Make a single prediction based on input fields"""
        try:
            # Gather input values
            features = {}
            for feature, var in self.input_vars.items():
                value = var.get().strip()
                if not value:
                    messagebox.showwarning("Input Error", 
                                         f"Please enter a value for {feature}")
                    return
                
                if feature not in ['Neighborhood']:
                    try:
                        value = float(value)
                    except ValueError:
                        messagebox.showwarning("Input Error", 
                                             f"{feature} must be a number")
                        return
                
                features[feature] = value
            
            # Make prediction
            predicted_price = self.predictor.predict_price(features)
            
            if predicted_price is not None:
                # Update result label
                result_text = f"Predicted Price: ₹{predicted_price * 75:,.2f}"
                self.result_label.configure(text=result_text)
                
                # Add to history and update plot
                self.prediction_history.append(predicted_price)
                self.update_plot()
            else:
                messagebox.showerror("Prediction Error", 
                                   "Failed to make prediction")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def clear_fields(self):
        """Clear all input fields"""
        for var in self.input_vars.values():
            var.set("")
        self.result_label.configure(text="")
    
    def load_csv(self):
        """Load house features from a CSV file and make batch predictions"""
        try:
            filename = filedialog.askopenfilename(
                filetypes=[("CSV files", "*.csv")]
            )
            if not filename:
                return
            
            # Load CSV
            df = pd.read_csv(filename)
            
            # Verify required columns
            missing_cols = set(self.feature_info.keys()) - set(df.columns)
            if missing_cols:
                messagebox.showerror("Error", 
                    f"Missing required columns: {', '.join(missing_cols)}")
                return
            
            # Make predictions
            results = []
            for _, row in df.iterrows():
                features = row[self.feature_info.keys()].to_dict()
                price = self.predictor.predict_price(features)
                if price is not None:
                    results.append(price)
            
            if results:
                # Save results
                save_path = filedialog.asksaveasfilename(
                    defaultextension=".csv",
                    filetypes=[("CSV files", "*.csv")]
                )
                if save_path:
                    df['Predicted_Price'] = results
                    df.to_csv(save_path, index=False)
                    messagebox.showinfo("Success", 
                        f"Predictions saved to {save_path}")
                
                # Update plot with batch results
                self.prediction_history.extend(results)
                self.update_plot()
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def update_plot(self):
        """Update the visualization plot"""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        if self.prediction_history:
            # Convert prediction history to INR
            inr_history = [p * 75 for p in self.prediction_history]
            # Create prediction history plot
            ax.plot(range(1, len(inr_history) + 1), 
                   inr_history, 'b-o')
            ax.set_xlabel('Prediction Number')
            ax.set_ylabel('Predicted Price (₹)')
            ax.set_title('Prediction History')
            ax.grid(True)
            
            # Format y-axis to show INR currency
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, p: f'₹{x:,.0f}'))
            
            # Rotate x-axis labels if there are many predictions
            if len(inr_history) > 10:
                plt.xticks(rotation=45)
            
            self.fig.tight_layout()
            self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = ModernHousePriceGUI(root)
    root.mainloop() 