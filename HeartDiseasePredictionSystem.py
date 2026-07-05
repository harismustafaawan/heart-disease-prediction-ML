import tkinter as tk
from tkinter import messagebox, filedialog, simpledialog, ttk
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
import threading
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ---------------- Global state ----------------
df = None
feature_names = []
target_name = None
X = None
y = None
scaler = None
models = {}
best_model = None
best_model_name = None
prediction = None
input_entries = []
models_trained = False
current_dataset_features = []  # Store features from loaded dataset
scaler_fitted = False  # Track if scaler is fitted to data

# ---------------- Utilities ----------------
def detect_target_column(dataframe):
    possible = ["target", "Target", "TARGET", "heartdisease", "HeartDisease", "heart_disease", "output"]
    for p in possible:
        if p in dataframe.columns:
            return p
    return dataframe.columns[-1]

def safe_float(x):
    try:
        return float(x)
    except:
        return np.nan

# ---------------- Dataset / Preprocess ----------------
def load_dataset():
    global df, feature_names, target_name, X, y, models_trained, current_dataset_features
    path = filedialog.askopenfilename(title="Select CSV file",
                                       filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
    if not path:
        return
    try:
        df = pd.read_csv(path)
    except Exception as e:
        messagebox.showerror("Load Error", f"Failed to read CSV:\n{e}")
        return

    target = detect_target_column(df)
    target_name_holder.set(f"Detected target: {target}")
    feature_names_local = [c for c in df.columns if c != target]

    # Store original feature names
    current_dataset_features = feature_names_local.copy()

    feature_names.clear()
    feature_names.extend(feature_names_local)
    X = df[feature_names].copy()
    y = df[target].copy()
    models_trained = False  # Reset training flag when new dataset loaded

    info = f"File loaded: {os.path.basename(path)}\nRows: {df.shape[0]}, Columns: {df.shape[1]}\nDetected target: {target}"
    output_text.configure(state="normal")
    output_text.delete(1.0, tk.END)
    output_text.insert(tk.END, info + "\n\n✔ Dataset loaded. Now click 'Preprocess & Prepare'\n")
    output_text.configure(state="disabled")
    build_input_fields()

def preprocess_and_run():
    global df, feature_names, target_name, X, y, scaler, models_trained, scaler_fitted
    if df is None:
        messagebox.showerror("Error", "Load dataset first.")
        return

    output_text.configure(state="normal")
    output_text.insert(tk.END, "\nPreprocessing started...\n")
    output_text.configure(state="disabled")
    root.update()

    df_clean = df.copy()

    # Convert object columns to numeric where possible
    for col in df_clean.columns:
        if df_clean[col].dtype == object:
            try:
                df_clean[col] = pd.to_numeric(df_clean[col].str.strip(), errors='coerce')
            except:
                pass

    # Encode remaining object columns
    for col in df_clean.columns:
        if df_clean[col].dtype == object:
            df_clean[col] = df_clean[col].astype('category').cat.codes

    # Handle missing values
    for col in df_clean.columns:
        if pd.api.types.is_numeric_dtype(df_clean[col]):
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        else:
            if not df_clean[col].mode().empty:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode().iloc[0])
            else:
                df_clean[col] = df_clean[col].fillna(0)

    target_name_local = detect_target_column(df_clean)
    feature_names.clear()
    feature_names.extend([c for c in df_clean.columns if c != target_name_local])
    X = df_clean[feature_names].copy()
    y = df_clean[target_name_local].copy()

    # Initialize and fit scaler
    scaler = StandardScaler()
    try:
        X_scaled = scaler.fit_transform(X)
        scaler_fitted = True
        output_text.configure(state="normal")
        output_text.insert(tk.END, f"✔ Preprocessing completed.\n")
        output_text.insert(tk.END, f"✔ Features: {len(feature_names)}\n")
        output_text.insert(tk.END, f"✔ Scaler fitted. Ready for training.\n")
        output_text.configure(state="disabled")
    except Exception as e:
        messagebox.showerror("Error", f"Scaling failed: {e}")
        return

    build_input_fields()
    root.update()

# ---------------- Training ----------------
def train_all_models():
    global X, y, scaler, models, best_model, best_model_name, models_trained, scaler_fitted
    if X is None or y is None:
        messagebox.showerror("Error", "Preprocess dataset first.")
        return

    if not scaler_fitted:
        messagebox.showerror("Error", "Scaler not fitted. Click 'Preprocess & Prepare' first.")
        return

    output_text.configure(state="normal")
    output_text.insert(tk.END, "\nTraining models...\n")
    output_text.configure(state="disabled")
    root.update()

    # Scale features
    X_scaled = scaler.transform(X)

    # Train-test split
    X_train_local, X_test_local, y_train_local, y_test_local = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Models dictionary
    model_dict = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "SVM (RBF)": SVC(kernel="rbf", probability=True)
    }

    results = []
    models.clear()

    # Check if target is multiclass or binary
    is_multiclass = len(np.unique(y)) > 2
    avg_type = 'weighted' if is_multiclass else 'binary'

    for name, mdl in model_dict.items():
        try:
            mdl.fit(X_train_local, y_train_local)
            preds = mdl.predict(X_test_local)

            # Metrics
            acc = accuracy_score(y_test_local, preds)
            prec = precision_score(y_test_local, preds, average=avg_type, zero_division=0)
            rec = recall_score(y_test_local, preds, average=avg_type, zero_division=0)
            f1 = f1_score(y_test_local, preds, average=avg_type, zero_division=0)

            results.append((name, acc, prec, rec, f1, mdl, X_test_local, y_test_local))
            models[name] = mdl

            output_text.configure(state="normal")
            output_text.insert(tk.END, f"✔ {name}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}\n")
            output_text.configure(state="disabled")
            root.update()
        except Exception as e:
            output_text.configure(state="normal")
            output_text.insert(tk.END, f"✘ Failed {name}: {e}\n")
            output_text.configure(state="disabled")

    # Save the best model
    if results:
        best = max(results, key=lambda x: x[4])  # Max F1
        best_model_name = best[0]
        best_model = models[best_model_name]

        try:
            # Also save feature names for later use
            with open("best_model.pkl", "wb") as f:
                pickle.dump(best_model, f)
            with open("best_scaler.pkl", "wb") as f:
                pickle.dump(scaler, f)
            with open("feature_names.pkl", "wb") as f:
                pickle.dump(feature_names, f)

            models_trained = True

            output_text.configure(state="normal")
            output_text.insert(tk.END, f"\n🏆 Best model: {best_model_name} (F1={best[4]:.4f})\n")
            output_text.insert(tk.END, f"✔ Saved: best_model.pkl, best_scaler.pkl, feature_names.pkl\n")
            output_text.insert(tk.END, f"✔ Ready for predictions!\n")
            output_text.configure(state="disabled")
        except Exception as e:
            messagebox.showwarning("Save Warning", f"Model trained but saving failed: {e}")
            models_trained = True  # Still trained even if save fails

        # store last training artifacts for plotting
        root._last_training_results = results  # attach to root for plotting use

# Thread safe wrapper for training
def train_thread():
    train_all_models()
    root.after(100, lambda: messagebox.showinfo("Training Complete", "Models trained successfully!\nYou can now apply regression."))

# ---------------- Demo model (fixed scoping) ----------------
def create_demo_models():
    global best_model, scaler, models_trained, feature_names, scaler_fitted, best_model_name
    try:
        # Create demo dataset similar to heart disease data
        np.random.seed(42)
        n_samples = 300
        # Typical heart disease features
        age = np.random.randint(30, 80, n_samples)
        chol = np.random.randint(150, 350, n_samples)
        thalach = np.random.randint(80, 200, n_samples)
        oldpeak = np.random.uniform(0, 5, n_samples)
        trestbps = np.random.randint(100, 180, n_samples)
        fbs = np.random.randint(0, 2, n_samples)
        restecg = np.random.randint(0, 3, n_samples)
        exang = np.random.randint(0, 2, n_samples)
        # Create target (1 = disease, 0 = no disease)
        target = ((age > 60) & (chol > 240) & (oldpeak > 2)).astype(int)

        # Create feature names (set global properly)
        feature_names.clear()
        feature_names.extend(['age', 'chol', 'thalach', 'oldpeak', 'trestbps', 'fbs', 'restecg', 'exang'])
        X_demo = np.column_stack([age, chol, thalach, oldpeak, trestbps, fbs, restecg, exang])
        y_demo = target

        # Scale features
        scaler_demo = StandardScaler()
        X_scaled = scaler_demo.fit_transform(X_demo)

        # Train model
        model_demo = LogisticRegression()
        model_demo.fit(X_scaled, y_demo)

        # Save everything
        with open("best_model.pkl", "wb") as f:
            pickle.dump(model_demo, f)
        with open("best_scaler.pkl", "wb") as f:
            pickle.dump(scaler_demo, f)
        with open("feature_names.pkl", "wb") as f:
            pickle.dump(feature_names, f)

        # Update global variables
        best_model = model_demo
        scaler = scaler_demo
        best_model_name = "Demo Heart Disease Model"
        models_trained = True
        scaler_fitted = True

        output_text.configure(state="normal")
        output_text.insert(tk.END, "✔ Created demo heart disease model\n")
        output_text.insert(tk.END, "Features: age, chol, thalach, oldpeak, trestbps, fbs, restecg, exang\n")
        output_text.configure(state="disabled")
        build_input_fields()

        # Auto-fill some example values
        example_values = [55, 220, 150, 1.2, 130, 1, 1, 0]
        for i, entry in enumerate(input_entries):
            if i < len(feature_names) and i < len(example_values):
                entry.delete(0, tk.END)
                entry.insert(0, str(example_values[i]))
                entry.config(fg="black")

        messagebox.showinfo("Demo Created",
                             "Demo heart disease model created!\n\n"
                             "Example values are pre-filled.\n"
                             "Click 'Apply Regression' to test prediction.")
    except Exception as e:
        messagebox.showerror("Demo Failed", f"Failed to create demo: {str(e)[:200]}")

# ---------------- Prediction ----------------
def apply_regression():
    global best_model, best_model_name, scaler, prediction, models_trained, scaler_fitted, feature_names

    # If model or scaler not present, try load from disk
    if not models_trained or best_model is None or scaler is None:
        try:
            with open("best_model.pkl", "rb") as f:
                best_model = pickle.load(f)
            with open("best_scaler.pkl", "rb") as f:
                scaler = pickle.load(f)
            with open("feature_names.pkl", "rb") as f:
                saved_features = pickle.load(f)

            # Ensure GUI uses the saved feature set (replace current names)
            if saved_features is None:
                saved_features = []

            # Overwrite feature_names to match the trained model exactly
            feature_names.clear()
            feature_names.extend(saved_features)

            # Rebuild input fields to match these names
            build_input_fields()
            root.update()

            best_model_name = "Loaded Model"
            models_trained = True
            scaler_fitted = True

            output_text.configure(state="normal")
            output_text.insert(tk.END, "✔ Loaded trained model from disk and rebuilt input fields to match.\n")
            output_text.configure(state="disabled")

        except FileNotFoundError:
            messagebox.showerror(
                "No Trained Models",
                "No trained models found!\n\n"
                "You need to:\n"
                "1. Load dataset & Preprocess\n"
                "2. Train All Models\nOR\nClick 'Create Demo Models' for testing"
            )
            return
        except Exception as e:
            messagebox.showerror("Load Error", f"Cannot load models: {e}")
            return

    # Ensure input fields match feature_names; if mismatch, rebuild
    if len(input_entries) != len(feature_names):
        build_input_fields()
        root.update()

    # Check again
    if not input_entries or len(input_entries) != len(feature_names):
        messagebox.showerror("Input Error", "Input fields not initialized correctly. Please reload or reset.")
        return

    # Collect values from input fields
    values = []
    missing_fields = []

    for i, (fname, ent) in enumerate(zip(feature_names, input_entries)):
        vstr = ent.get().strip()
        # treat placeholder as empty
        if vstr == "" or vstr.lower() == "enter value...":
            missing_fields.append(fname)
            # Try to get median from dataset if available
            if df is not None and fname in df.columns:
                median_val = df[fname].median()
                vstr = str(median_val)
                ent.delete(0, tk.END)
                ent.insert(0, vstr)
                ent.config(fg="black")
            else:
                # Prompt the user for missing value (but this rarely happens after auto rebuild)
                vstr = simpledialog.askstring("Missing Value",
                                               f"Enter value for '{fname}':\n(Feature {i+1}/{len(feature_names)})")
                if vstr is None:
                    messagebox.showerror("Input Cancelled", "Prediction cancelled.")
                    return

        v = safe_float(vstr)
        if np.isnan(v):
            messagebox.showerror("Invalid Input", f"Invalid number for '{fname}': {vstr}")
            return
        values.append(v)

    # Show warning for missing values that were auto-filled
    if missing_fields and df is not None:
        output_text.configure(state="normal")
        output_text.insert(tk.END, f"Auto-filled missing values for: {', '.join(missing_fields[:5])}\n")
        output_text.configure(state="disabled")

    # Make prediction
    try:
        arr = np.array(values).reshape(1, -1)

        # Final safety check: scaler expected feature count
        expected = getattr(scaler, "n_features_in_", None)
        if expected is not None and arr.shape[1] != expected:
            messagebox.showerror("Dimension Error",
                                  f"Feature mismatch!\n\nExpected {expected} features (model/scaler), got {arr.shape[1]}.\n"
                                  f"To fix: Ensure you built/loaded the correct model for this dataset.")
            return

        arr_scaled = scaler.transform(arr)
        pred = best_model.predict(arr_scaled)[0]
        prediction = int(pred)

        # Get probabilities if available
        prob_str = ""
        if hasattr(best_model, "predict_proba"):
            prob = best_model.predict_proba(arr_scaled)[0]
            prob_str = f"\nProbabilities: {np.round(prob, 3)}"

        # Show detailed prediction
        output_text.configure(state="normal")
        output_text.insert(tk.END, f"\n🔍 PREDICTION RESULTS:\n")
        output_text.insert(tk.END, f"Model: {best_model_name}\n")
        output_text.insert(tk.END, f"Prediction: {prediction}{prob_str}\n")

        # Show input summary
        output_text.insert(tk.END, f"\nInput Values:\n")
        for i, (fname, val) in enumerate(zip(feature_names[:5], values[:5])):
            output_text.insert(tk.END, f"  {fname}: {val:.2f}\n")
        if len(feature_names) > 5:
            output_text.insert(tk.END, f"  ... and {len(feature_names)-5} more features\n")

        output_text.configure(state="disabled")

        # Scroll to bottom
        output_text.see(tk.END)

        messagebox.showinfo("Prediction Complete",
                             f"Prediction: {prediction}\n\n"
                             f"Model: {best_model_name}\n"
                             f"Now click 'Show Result' to see interpretation.")

    except Exception as e:
        error_msg = str(e)
        if "dimensions" in error_msg.lower() or "shape" in error_msg.lower():
            expected = getattr(scaler, "n_features_in_", "unknown")
            messagebox.showerror("Dimension Error",
                                  f"Feature mismatch!\n\n"
                                  f"Expected {expected} features, got {len(values)}\n"
                                  f"Please train models with current dataset or load appropriate model.")
        elif "fitted" in error_msg.lower():
            messagebox.showerror("Scaler Error",
                                  "Scaler not properly fitted to data.\n"
                                  "Please click 'Preprocess & Prepare' then 'Train All Models'.")
        else:
            messagebox.showerror("Prediction Error", f"Failed to predict: {error_msg}")

# ---------------- Show / Interpret ----------------
def show_result():
    global prediction, best_model
    if prediction is None:
        messagebox.showerror("Error", "No prediction found. Click 'Apply Regression' first.")
        return

    # Get class information from model
    if hasattr(best_model, 'classes_'):
        classes = best_model.classes_

        # Binary classification (most common for heart disease)
        if len(classes) == 2:
            # Determine which class represents disease (try 1)
            disease_class = 1 if 1 in classes else classes[-1]
            if prediction == disease_class:
                result_text = "⚠ HEART DISEASE DETECTED\n\n" \
                              "Recommendations:\n" \
                              "• Consult a cardiologist immediately\n" \
                              "• Monitor blood pressure regularly\n" \
                              "• Follow a heart-healthy diet\n" \
                              "• Exercise regularly\n" \
                              "• Avoid smoking and excessive alcohol"
            else:
                result_text = "✔ NO HEART DISEASE DETECTED\n\n" \
                              "Your heart health appears normal.\n" \
                              "Continue maintaining a healthy lifestyle!"
        else:
            # Multiclass classification
            result_text = f"Prediction: Class {prediction}\n\n" \
                          f"Possible classes: {list(classes)}"
    else:
        # Fallback for unknown models
        if prediction == 1:
            result_text = "⚠ Heart Disease Detected\nPlease consult a doctor."
        else:
            result_text = "✔ No Heart Disease Detected\nYour heart appears healthy."

    messagebox.showinfo("Diagnosis Result", result_text)

# ---------------- Program preview ----------------
def load_program():
    path = filedialog.askopenfilename(title="Select a Python program (optional)",
                                       filetypes=[("Python files", "*.py"), ("All files", "*.*")])
    if not path:
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            txt = "".join(f.readlines()[:200])
        win = tk.Toplevel(root)
        win.title("Program Preview: " + os.path.basename(path))
        txtw = tk.Text(win, width=100, height=30)
        txtw.pack(fill="both", expand=True)
        txtw.insert(tk.END, txt)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to open file: {e}")

# ---------------- Build Input Fields ----------------
def build_input_fields():
    for w in fields_container.winfo_children():
        w.destroy()
    input_entries.clear()

    if not feature_names:
        tk.Label(fields_container,
                 text="No features available.\nLoad dataset or create demo models.",
                 fg="gray", font=("Arial", 10)).pack(expand=True, pady=50)
        return

    # Create scrollable frame
    canvas = tk.Canvas(fields_container, bg="white")
    scrollbar = ttk.Scrollbar(fields_container, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)

    scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    # Create input fields
    for i, fname in enumerate(feature_names):
        row = tk.Frame(scrollable_frame)
        row.pack(fill="x", pady=2, padx=4)

        lbl = tk.Label(row, text=f"{i+1}. {fname}", width=25, anchor="w", font=("Arial", 9))
        lbl.pack(side="left")

        ent = tk.Entry(row, width=20, font=("Arial", 9))
        ent.pack(side="right", fill="x", expand=True, padx=(10, 0))

        # Add placeholder text
        ent.insert(0, "Enter value...")
        ent.config(fg="grey")

        def on_entry_click(event, entry=ent, default_text="Enter value..."):
            if entry.get() == default_text:
                entry.delete(0, tk.END)
                entry.config(fg="black")

        def on_focusout(event, entry=ent, default_text="Enter value..."):
            if entry.get() == '':
                entry.insert(0, default_text)
                entry.config(fg="grey")

        ent.bind('<FocusIn>', on_entry_click)
        ent.bind('<FocusOut>', on_focusout)

        input_entries.append(ent)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # Add note
    note_frame = tk.Frame(scrollable_frame)
    note_frame.pack(fill="x", pady=10)
    tk.Label(note_frame,
             text="💡 Tip: Leave blank to use median values from dataset",
             fg="#349808", font=("Arial", 9)).pack()

# ---------------- Plotting ----------------
def show_graphs():
    """
    Plots:
    - Confusion matrix for best model (if test set exists)
    - Feature importance (if RandomForest trained)
    - ROC curve (if binary)
    Uses artifacts stored in root._last_training_results by train_all_models.
    """
    results = getattr(root, "_last_training_results", None)
    if not results:
        messagebox.showerror("No Data", "No training results available. Train models first.")
        return

    # take the tuple for best model if present
    best_tuple = None
    for t in results:
        if t[0] == best_model_name:
            best_tuple = t
            break
    if best_tuple is None:
        # pick first result
        best_tuple = results[0]

    name, acc, prec, rec, f1, mdl, X_test_local, y_test_local = best_tuple

    # Create plot window
    win = tk.Toplevel(root)
    win.title("Model Graphs")
    win.geometry("900x700")

    fig = Figure(figsize=(8.5, 7))
    axs = fig.subplots(3, 1)
    fig.tight_layout(pad=3.0)

    # 1) Confusion Matrix
    try:
        preds = mdl.predict(X_test_local)
        cm = confusion_matrix(y_test_local, preds)
        axs[0].set_title(f"Confusion Matrix - {name}")
        im = axs[0].imshow(cm, interpolation='nearest', aspect='auto')
        for (i, j), val in np.ndenumerate(cm):
            axs[0].text(j, i, str(val), ha='center', va='center', color='white' if cm.max() > 0 else 'black')
        axs[0].set_ylabel('True')
        axs[0].set_xlabel('Predicted')
    except Exception as e:
        axs[0].text(0.1, 0.5, f"Confusion matrix unavailable:\n{e}", fontsize=10)
        axs[0].axis('off')

    # 2) Feature importance (if RandomForest)
    if "Random Forest" in models:
        try:
            rf = models["Random Forest"]
            if hasattr(rf, "feature_importances_"):
                imp = rf.feature_importances_
                sorted_idx = np.argsort(imp)[::-1]
                topk = min(len(imp), 12)
                names = [feature_names[i] for i in sorted_idx[:topk]]
                vals = imp[sorted_idx[:topk]]
                axs[1].barh(range(len(vals))[::-1], vals, edgecolor='k')
                axs[1].set_yticks(range(len(vals)))
                axs[1].set_yticklabels(names[::-1])
                axs[1].set_title("Random Forest Feature Importances (top features)")
                axs[1].set_xlabel("Importance")
            else:
                axs[1].text(0.1, 0.5, "Random Forest has no feature_importances_", fontsize=10)
                axs[1].axis('off')
        except Exception as e:
            axs[1].text(0.1, 0.5, f"Feature importance unavailable:\n{e}", fontsize=10)
            axs[1].axis('off')
    else:
        axs[1].text(0.1, 0.5, "Random Forest model not available\n(Train models to enable feature importance)", fontsize=10)
        axs[1].axis('off')

    # 3) ROC curve (only for binary)
    try:
        classes = getattr(mdl, "classes_", None)
        if classes is not None and len(classes) == 2 and hasattr(mdl, "predict_proba"):
            probs = mdl.predict_proba(X_test_local)[:, 1]
            fpr, tpr, _ = roc_curve(y_test_local, probs)
            roc_auc = auc(fpr, tpr)
            axs[2].plot(fpr, tpr, lw=2)
            axs[2].plot([0, 1], [0, 1], linestyle='--', lw=1)
            axs[2].set_title(f"ROC Curve - {name} (AUC = {roc_auc:.3f})")
            axs[2].set_xlabel("False Positive Rate")
            axs[2].set_ylabel("True Positive Rate")
        else:
            axs[2].text(0.1, 0.5, "ROC curve not available (requires binary classification & predict_proba)", fontsize=10)
            axs[2].axis('off')
    except Exception as e:
        axs[2].text(0.1, 0.5, f"ROC unavailable:\n{e}", fontsize=10)
        axs[2].axis('off')

    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

# ---------------- GUI ----------------
root = tk.Tk()
root.title("Heart Disease ML System")
root.geometry("1100x750")

# Configure style
style = ttk.Style()
style.configure("TButton", padding=6, relief="flat")

header = tk.Frame(root, pady=8, bg="#2C3E50")
header.pack(fill="x")
tk.Label(header, text="❤ Heart Disease Prediction System",
         font=("Helvetica", 20, "bold"), bg="#2C3E50", fg="white").pack()

top = tk.Frame(root)
top.pack(fill="both", expand=False, padx=10, pady=5)

btns = tk.Frame(top)
btns.pack(side="left", fill="y", padx=10)

# Button colors
colors = {
    "load": "#3498DB",
    "preprocess": "#F39C12",
    "train": "#27AE60",
    "demo": "#9B59B6",
    "predict": "#E74C3C",
    "result": "#C0392B",
    "program": "#7F8C8D",
    "graphs": "#16A085"
}

tk.Button(btns, text="1. Load Dataset", width=28, command=load_dataset,
          bg=colors["load"], fg="white", font=("Arial", 10, "bold")).pack(pady=5)
tk.Button(btns, text="2. Preprocess & Prepare", width=28, command=preprocess_and_run,
          bg=colors["preprocess"], fg="white", font=("Arial", 10, "bold")).pack(pady=5)
tk.Button(btns, text="3. Train All Models", width=28,
          command=lambda: threading.Thread(target=train_thread).start(),
          bg=colors["train"], fg="white", font=("Arial", 10, "bold")).pack(pady=5)
tk.Button(btns, text="Create Demo Models", width=28, command=create_demo_models,
          bg=colors["demo"], fg="white", font=("Arial", 10, "bold")).pack(pady=5)
tk.Button(btns, text="4. Apply Regression (Predict)", width=28, command=apply_regression,
          bg=colors["predict"], fg="white", font=("Arial", 10, "bold")).pack(pady=5)
tk.Button(btns, text="5. Show Result", width=28, command=show_result,
          bg=colors["result"], fg="white", font=("Arial", 10, "bold")).pack(pady=5)
tk.Button(btns, text="Show Graphs", width=28, command=show_graphs,
          bg=colors["graphs"], fg="white", font=("Arial", 10, "bold")).pack(pady=8)
tk.Button(btns, text="Load Program (optional)", width=28, command=load_program,
          bg=colors["program"], fg="white", font=("Arial", 10)).pack(pady=5)

target_name_holder = tk.StringVar(value="No dataset loaded.")
tk.Label(btns, textvariable=target_name_holder, wraplength=260,
         justify="left", bg="#ECF0F1", relief="solid", padx=10, pady=5).pack(pady=10, fill="x")

out_frame = tk.Frame(top)
out_frame.pack(side="right", fill="both", expand=True)

tk.Label(out_frame, text="Output / Logs:", font=("Arial", 12, "bold")).pack(anchor="w")
output_text = tk.Text(out_frame, width=80, height=18, bg="#F8F9F9", relief="solid")
output_text.pack(fill="both", expand=True)
output_text.configure(state="disabled")

# Add scrollbar to output
output_scroll = ttk.Scrollbar(out_frame, orient="vertical", command=output_text.yview)
output_text.configure(yscrollcommand=output_scroll.set)
output_scroll.pack(side="right", fill="y")

middle = tk.Frame(root)
middle.pack(fill="both", expand=True, padx=10, pady=6)

input_frame = tk.Frame(middle, bd=2, relief="groove")
input_frame.pack(side="left", fill="both", expand=True, padx=6, pady=6)

tk.Label(input_frame, text="Input Fields for Prediction:", font=("Arial", 12, "bold")).pack(anchor="w", padx=4, pady=4)
fields_container = tk.Frame(input_frame)
fields_container.pack(fill="both", expand=True, padx=4, pady=4)

build_input_fields()  # initial call (no features yet)

right = tk.Frame(middle, width=320, relief="solid", bd=1)
right.pack(side="right", fill="y", padx=6)
tk.Label(right, text="📋 Instructions", font=("Arial", 12, "bold"), bg="#ECF0F1").pack(fill="x", pady=5)

instructions_text = """FOR REAL DATASET:

1. LOAD DATASET
   • Click button
   • Select your CSV file
   • Target column auto-detected

2. PREPROCESS
   • Click button
   • Data cleaned & scaled
   • Features extracted

3. TRAIN MODELS
   • Click button (wait)
   • 3 models trained
   • Best model auto-selected

4. FILL INPUTS
   • Enter values
   • or leave blank for auto-fill

5. APPLY REGRESSION
   • Get prediction

6. SHOW RESULT
   • See diagnosis

To see graphs:
After training, click "Show Graphs"
(Confusion matrix, feature importance, ROC)"""

preview_text = tk.Text(right, width=40, height=22, bg="#FDFEFE", relief="flat")
preview_text.pack(fill="both", expand=True, pady=5)
preview_text.insert(tk.END, instructions_text)
preview_text.configure(state="disabled", font=("Arial", 9))

footer = tk.Frame(root, pady=6, bg="#34495E")
footer.pack(fill="x", side="bottom")

root.mainloop()
