import re

from flask import (
    Flask,
    make_response,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    jsonify,
)

import pickle
import pandas as pd
import numpy as np
import os
from datetime import datetime

from dotenv import load_dotenv

from utils.prediction import (
    prepare_sale_input,
    prepare_rental_input
)

from flask_login import (
    LoginManager,
    login_user,
    logout_user,
    login_required,
    current_user
)

from models import db, User, Valuation, InsuranceAssessment
from auth_decorators import role_required

from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
)

# ── Environment ───────────────────────────────────────────────────────────────

load_dotenv()

app = Flask(__name__)

app.config["SECRET_KEY"]                  = os.getenv("SECRET_KEY")
app.config["SQLALCHEMY_DATABASE_URI"]     = os.getenv("DATABASE_URL")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)

with app.app_context():
    db.create_all()

# ── Flask-Login ───────────────────────────────────────────────────────────────

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ── Dataset loading ───────────────────────────────────────────────────────────

DATASET_PATH = "dataset/kenya_properties.csv"

full_df = pd.read_csv(DATASET_PATH)

sale_location_df   = full_df[full_df["listing_type"] == "For Sale"].copy()
rental_location_df = full_df[full_df["listing_type"] == "For Rent"].copy()

# Location + neighborhood factor lookup (neighborhood-level, not property-level)
FACTOR_COLS = [
    "neighborhood", "distance_to_cbd_km", "location_tier",
    "distance_to_school_km", "distance_to_hospital_km",
    "crime_index", "road_quality",
]

sale_location_lookup = (
    sale_location_df[["neighborhood", "town"] + FACTOR_COLS[1:]]
    .drop_duplicates(subset=["neighborhood", "town"])
)

rental_location_lookup = (
    rental_location_df[["neighborhood", "town"] + FACTOR_COLS[1:]]
    .drop_duplicates(subset=["neighborhood", "town"])
)

# ── Model loading ─────────────────────────────────────────────────────────────

with open("models/sale_model.pkl",      "rb") as f: sale_model      = pickle.load(f)
with open("models/sale_encoders.pkl",   "rb") as f: sale_encoders   = pickle.load(f)
with open("models/rental_model.pkl",    "rb") as f: rental_model    = pickle.load(f)
with open("models/rental_encoders.pkl", "rb") as f: rental_encoders = pickle.load(f)

# ── PDF report builder ────────────────────────────────────────────────────────

BROWN_DARK = colors.HexColor("#1C140D")
BROWN_MID  = colors.HexColor("#6B4A30")
GOLD       = colors.HexColor("#C9A24B")

PROPERTY_LABELS = {
    "neighborhood":           "Neighborhood",
    "town":                   "Town",
    "property_type":          "Property Type",
    "bedrooms":               "Bedrooms",
    "bathrooms":              "Bathrooms",
    "floor_size_sqm":         "Floor Size",
    "year_built":             "Year Built",
    "condition":              "Condition",
    "furnishing":             "Furnishing",
    "distance_to_cbd_km":     "Distance to CBD",
    "parking_spaces":         "Parking Spaces",
    "floor_number":           "Floor Number",
    "distance_to_school_km":  "Distance to School",
    "distance_to_hospital_km":"Distance to Hospital",
    "crime_index":            "Crime Index",
    "road_quality":           "Road Quality",
}

def build_pdf_report(report_title, subtitle, price_label, price_value,
                     property_args, extra_sections=None):

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=22*mm, bottomMargin=22*mm,
        leftMargin=22*mm, rightMargin=22*mm,
    )
    styles = getSampleStyleSheet()

    def style(name, **kw):
        return ParagraphStyle(name, parent=styles["Normal"], **kw)

    title_s    = style("T", fontName="Helvetica-Bold",    fontSize=20,  textColor=BROWN_DARK, alignment=TA_LEFT, spaceAfter=2)
    sub_s      = style("S", fontName="Helvetica",         fontSize=10,  textColor=BROWN_MID,  spaceAfter=16)
    plabel_s   = style("PL", fontName="Helvetica",        fontSize=10,  textColor=colors.white, spaceAfter=4)
    pvalue_s   = style("PV", fontName="Helvetica-Bold",   fontSize=24,  textColor=colors.white)
    sec_s      = style("H", fontName="Helvetica-Bold",    fontSize=12,  textColor=BROWN_DARK, spaceBefore=16, spaceAfter=10)
    disc_s     = style("D", fontName="Helvetica-Oblique", fontSize=8.5, textColor=BROWN_MID, leading=12)
    foot_s     = style("F", fontName="Helvetica",         fontSize=8.5, textColor=BROWN_MID)

    story = [
        Paragraph(report_title, title_s),
        Paragraph(subtitle, sub_s),
        HRFlowable(width="100%", thickness=1, color=GOLD, spaceAfter=16),
    ]

    price_table = Table(
        [[Paragraph(price_label, plabel_s)],
         [Paragraph(f"KES {price_value}", pvalue_s)]],
        colWidths=[doc.width],
    )
    price_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), BROWN_DARK),
        ("LEFTPADDING",   (0,0), (-1,-1), 16),
        ("RIGHTPADDING",  (0,0), (-1,-1), 16),
        ("TOPPADDING",    (0,0), (-1, 0), 14),
        ("BOTTOMPADDING", (0,1), (-1,-1), 16),
        ("TOPPADDING",    (0,1), (-1,-1), 2),
    ]))
    story += [price_table, Spacer(1, 20), Paragraph("Property Details", sec_s)]

    rows = []
    for key, label in PROPERTY_LABELS.items():
        val = property_args.get(key)
        if val in (None, "", "None"): continue
        if key == "floor_size_sqm":     val = f"{val} sqm"
        elif key == "distance_to_cbd_km":    val = f"{val} km"
        elif key == "distance_to_school_km": val = f"{val} km"
        elif key == "distance_to_hospital_km": val = f"{val} km"
        rows.append([label, str(val)])

    if rows:
        dtable = Table(rows, colWidths=[doc.width*0.4, doc.width*0.6])
        dtable.setStyle(TableStyle([
            ("FONTNAME",      (0,0), (0,-1), "Helvetica-Bold"),
            ("FONTNAME",      (1,0), (1,-1), "Helvetica"),
            ("FONTSIZE",      (0,0), (-1,-1), 10.5),
            ("TEXTCOLOR",     (0,0), (0,-1), BROWN_MID),
            ("TEXTCOLOR",     (1,0), (1,-1), BROWN_DARK),
            ("BOTTOMPADDING", (0,0), (-1,-1), 8),
            ("TOPPADDING",    (0,0), (-1,-1), 8),
            ("LINEBELOW",     (0,0), (-1,-2), 0.5, colors.HexColor("#E7DDD0")),
        ]))
        story.append(dtable)

    if extra_sections:
        for heading, extra_rows in extra_sections:
            story.append(Paragraph(heading, sec_s))
            et = Table([[l, str(v)] for l, v in extra_rows],
                       colWidths=[doc.width*0.4, doc.width*0.6])
            et.setStyle(TableStyle([
                ("FONTNAME",      (0,0), (0,-1), "Helvetica-Bold"),
                ("FONTNAME",      (1,0), (1,-1), "Helvetica"),
                ("FONTSIZE",      (0,0), (-1,-1), 10.5),
                ("TEXTCOLOR",     (0,0), (0,-1), BROWN_MID),
                ("TEXTCOLOR",     (1,0), (1,-1), BROWN_DARK),
                ("BOTTOMPADDING", (0,0), (-1,-1), 8),
                ("TOPPADDING",    (0,0), (-1,-1), 8),
                ("LINEBELOW",     (0,0), (-1,-2), 0.5, colors.HexColor("#E7DDD0")),
            ]))
            story.append(et)

    story += [
        Spacer(1, 24),
        HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#E7DDD0"), spaceAfter=12),
        Paragraph(
            "This figure was generated using a machine learning model trained on "
            "property features, location, and market data across Kenya. It is an "
            "estimate, not a formal appraisal.", disc_s),
        Spacer(1, 16),
        Paragraph("Generated by PropAI", foot_s),
    ]

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ── Risk assessment helper ─────────────────────────────────────────────────────

def assess_risk(data, estimated_value):
    """
    Rule-based risk scoring using original (pre-encoding) string values.
    Receives display_data — human-readable strings, not label-encoded ints.
    """
    score   = 0
    factors = []

    # Building age
    try:
        age = 2026 - int(data.get("year_built", 2020))
    except (ValueError, TypeError):
        age = 6

    if age > 30:
        score += 3
        factors.append("Building is over 30 years old, increasing structural risk.")
    elif age > 15:
        score += 1
        factors.append("Building is moderately aged (15–30 years).")
    else:
        factors.append("Building is relatively new, reducing structural risk.")

    # Condition
    condition = data.get("condition", "")
    if condition in ("Excellent", "New"):
        score -= 1
        factors.append("Property condition is excellent, lowering risk.")
    elif condition == "Fair":
        score += 2
        factors.append("Property condition is only fair, raising risk.")

    # Security features (checkbox values arrive as "1" string)
    has_security = data.get("has_security") == "1"
    is_gated     = data.get("is_gated_community") == "1"
    has_gen      = data.get("has_backup_generator") == "1"
    has_borehole = data.get("has_borehole") == "1"

    if has_security:
        score -= 1
        factors.append("On-site security reduces risk of loss.")
    if is_gated:
        score -= 1
        factors.append("Gated community access reduces risk exposure.")
    if not has_security and not is_gated:
        score += 2
        factors.append("No security or gated access in place, raising risk.")
    if has_gen:
        score -= 1
        factors.append("Backup generator reduces risk from power-related damage.")
    if has_borehole:
        score -= 1
        factors.append("Borehole reduces dependency-related risk.")

    # Location tier (real string values from dataset)
    tier = str(data.get("location_tier", "mid")).strip().lower()
    if tier == "premium":
        score -= 1
        factors.append("Premium location tier reduces overall risk.")
    elif tier == "affordable":
        score += 2
        factors.append("Lower-tier (affordable) location raises overall risk.")

   # Crime index (new factor — higher index = higher risk)
    try:
        # Use float instead of int to handle decimal values
        crime = float(data.get("crime_index", 20))
    except (ValueError, TypeError):
        crime = 20

    if crime <= 10:
        score -= 1
        factors.append(f"Low crime index ({crime}) in this neighborhood reduces risk.")
    elif crime >= 35:
        score += 2
        factors.append(f"High crime index ({crime}) in this neighborhood raises risk.")
    else:
        factors.append(f"Moderate crime index ({crime}) in this neighborhood.")

        # Road quality (new factor)
    road = str(data.get("road_quality", "Paved")).strip()
    if road == "Murram":
        score += 1
        factors.append("Murram/unpaved roads increase accessibility and maintenance risk.")
    elif road == "Mixed":
        factors.append("Mixed road quality in this neighborhood.")
    else:
        factors.append("Paved roads in this neighborhood reduce accessibility risk.")

    # Distance to hospital (new factor)
    try:
        hosp_dist = float(data.get("distance_to_hospital_km", 2.0))
    except (ValueError, TypeError):
        hosp_dist = 2.0

    if hosp_dist > 5:
        score += 1
        factors.append(f"Distance to nearest hospital ({hosp_dist} km) may increase emergency response times.")
    elif hosp_dist <= 1.5:
        score -= 1
        factors.append(f"Close proximity to hospital ({hosp_dist} km) reduces emergency risk.")

    # Final tier
    if score <= 0:
        risk_tier     = "Low"
        coverage_low  = estimated_value * 0.85
        coverage_high = estimated_value * 1.0
    elif score <= 4:
        risk_tier     = "Medium"
        coverage_low  = estimated_value * 0.70
        coverage_high = estimated_value * 0.85
    else:
        risk_tier     = "High"
        coverage_low  = estimated_value * 0.50
        coverage_high = estimated_value * 0.70

    return risk_tier, factors, coverage_low, coverage_high


# ── Lookup helper ─────────────────────────────────────────────────────────────

def get_location_row(lookup_df, neighborhood):
    """Returns the first matching row as a dict, or None."""
    rows = lookup_df[lookup_df["neighborhood"] == neighborhood]
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

# ── Home (role-aware) ─────────────────────────────────────────────────────────

@app.route("/")
def home():
    if current_user.is_authenticated:
        if current_user.role == "insurance":
            return render_template("insurance_home.html")
        if current_user.role == "admin":
            return render_template("admin_home.html")
        if current_user.role == "owner":
            return render_template("index.html")
    return render_template("index.html")


# ── Auth ──────────────────────────────────────────────────────────────────────

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        full_name = request.form["full_name"].strip()
        email = request.form["email"].strip().lower()
        password = request.form["password"]
        role = request.form.get("role", "owner")

        # Validate full name
        if not re.fullmatch(r"[A-Za-zÀ-ÿ' -]+", full_name):
            flash("Name can only contain letters, spaces, hyphens and apostrophes.")
            return redirect(url_for("register"))

        # Validate password
        if len(password) < 8:
            flash("Password must be at least 8 characters long.")
            return redirect(url_for("register"))

        if not re.search(r"[A-Z]", password):
            flash("Password must contain at least one uppercase letter.")
            return redirect(url_for("register"))

        if not re.search(r"[a-z]", password):
            flash("Password must contain at least one lowercase letter.")
            return redirect(url_for("register"))

        if not re.search(r"[0-9]", password):
            flash("Password must contain at least one number.")
            return redirect(url_for("register"))

        if not re.search(r"[^A-Za-z0-9]", password):
            flash("Password must contain at least one special character.")
            return redirect(url_for("register"))

        # Check if email already exists
        if User.query.filter_by(email=email).first():
            flash("Email already exists.")
            return redirect(url_for("register"))

        # Create user
        user = User(full_name=full_name, email=email, role=role)
        user.set_password(password)

        db.session.add(user)
        db.session.commit()

        login_user(user)
        return redirect(url_for("home"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email    = request.form["email"]
        password = request.form["password"]
        user     = User.query.filter_by(email=email).first()

        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for("home"))

        flash("Invalid credentials.")
    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("home"))


# ── Owner dashboard ───────────────────────────────────────────────────────────

@app.route("/dashboard")
@login_required
@role_required("owner")
def owner_dashboard():

    recent = (Valuation.query
              .filter_by(user_id=current_user.id)
              .order_by(Valuation.created_at.desc())
              .limit(5).all())

    all_v       = Valuation.query.filter_by(user_id=current_user.id).all()
    sale_vals   = [v for v in all_v if v.valuation_type == "sale"]
    rental_vals = [v for v in all_v if v.valuation_type == "rental"]

    avg_sale = (sum(v.predicted_price for v in sale_vals) / len(sale_vals)
                if sale_vals else None)

    stats = {
        "total":        len(all_v),
        "sale_count":   len(sale_vals),
        "rental_count": len(rental_vals),
        "avg_sale_price": avg_sale,
    }

    # Market summary from dataset
    summary = []
    for town in sorted(full_df["town"].unique()):
        town_df     = full_df[full_df["town"] == town]
        sale_rows   = town_df[town_df["listing_type"] == "For Sale"]["price_ksh"]
        rental_rows = town_df[town_df["listing_type"] == "For Rent"]["price_ksh"]
        summary.append({
            "town":       town,
            "avg_sale":   sale_rows.mean()   if not sale_rows.empty   else None,
            "avg_rental": rental_rows.mean() if not rental_rows.empty else None,
            "count":      len(town_df),
        })

    return render_template(
        "owner_dashboard.html",
        recent_valuations=recent,
        stats=stats,
        market_summary=summary,
    )


# ── Insurance dashboard ───────────────────────────────────────────────────────

@app.route("/insurance/dashboard")
@login_required
@role_required("insurance")
def insurance_dashboard():

    recent = (InsuranceAssessment.query
              .filter_by(user_id=current_user.id)
              .order_by(InsuranceAssessment.created_at.desc())
              .limit(5).all())

    all_a = InsuranceAssessment.query.filter_by(user_id=current_user.id).all()

    stats = {
        "total":        len(all_a),
        "low_count":    sum(1 for a in all_a if a.risk_tier == "Low"),
        "medium_count": sum(1 for a in all_a if a.risk_tier == "Medium"),
        "high_count":   sum(1 for a in all_a if a.risk_tier == "High"),
    }

    return render_template(
        "insurance_dashboard.html",
        recent_assessments=recent,
        stats=stats,
    )


# ── Location API ──────────────────────────────────────────────────────────────

@app.route("/get_neighborhoods/<property_type>/<town>")
def get_neighborhoods(property_type, town):
    df = sale_location_df if property_type == "sale" else rental_location_df
    locations = (df[df["town"] == town]["neighborhood"]
                 .dropna().unique())
    return {"neighborhoods": list(locations)}


@app.route("/get_neighborhood_details/<property_type>/<town>")
@login_required
def get_neighborhood_details(property_type, town):
    lookup = sale_location_lookup if property_type == "sale" else rental_location_lookup
    df = sale_location_df if property_type == "sale" else rental_location_df

    town_neighborhoods = df[df["town"] == town]["neighborhood"].unique()
    matches = lookup[lookup["neighborhood"].isin(town_neighborhoods)]

    results = [
        {
            "neighborhood":           row["neighborhood"],
            "distance_to_cbd_km":     row["distance_to_cbd_km"],
            "location_tier":          row["location_tier"],
            "distance_to_school_km":  row.get("distance_to_school_km"),
            "distance_to_hospital_km":row.get("distance_to_hospital_km"),
            "crime_index":            row.get("crime_index"),
            "road_quality":           row.get("road_quality"),
        }
        for _, row in matches.iterrows()
    ]
    return {"neighborhoods": results}


# ── Sale valuation ────────────────────────────────────────────────────────────

@app.route("/sale", methods=["GET", "POST"])
@login_required
@role_required("owner")
def sale():
    if request.method == "POST":
        data     = request.form.to_dict()
        selected = data["neighborhood"]

        loc = get_location_row(sale_location_lookup, selected)
        if loc is None:
            flash("Invalid location selected.")
            return redirect(url_for("sale"))

        data["distance_to_cbd_km"]      = loc["distance_to_cbd_km"]
        data["location_tier"]           = loc["location_tier"]
        data["distance_to_school_km"]   = loc.get("distance_to_school_km")
        data["distance_to_hospital_km"] = loc.get("distance_to_hospital_km")
        data["crime_index"]             = loc.get("crime_index")
        data["road_quality"]            = loc.get("road_quality")

        display_data = data.copy()

        df = prepare_sale_input(data, sale_encoders)
        df = df.reindex(columns=sale_model.feature_names_in_, fill_value=0)

        prediction = round(sale_model.predict(df)[0])

        # Save to database
        v = Valuation(
            user_id              = current_user.id,
            label                = data.get("label") or None,
            valuation_type       = "sale",
            town                 = display_data["town"],
            neighborhood         = display_data["neighborhood"],
            distance_to_cbd_km   = display_data.get("distance_to_cbd_km"),
            location_tier        = display_data.get("location_tier"),
            property_type        = display_data["property_type"],
            bedrooms             = int(display_data.get("bedrooms", 0) or 0),
            bathrooms            = int(display_data.get("bathrooms", 0) or 0),
            floor_size_sqm       = int(display_data.get("floor_size_sqm", 0) or 0),
            year_built           = int(display_data.get("year_built", 2000) or 2000),
            parking_spaces       = int(display_data.get("parking_spaces", 0) or 0),
            floor_number         = int(display_data.get("floor_number", 0) or 0),
            condition            = display_data.get("condition"),
            furnishing           = display_data.get("furnishing"),
            has_swimming_pool    = int(display_data.get("has_swimming_pool", 0) or 0),
            has_gym              = int(display_data.get("has_gym", 0) or 0),
            has_borehole         = int(display_data.get("has_borehole", 0) or 0),
            has_backup_generator = int(display_data.get("has_backup_generator", 0) or 0),
            has_security         = int(display_data.get("has_security", 0) or 0),
            has_garden           = int(display_data.get("has_garden", 0) or 0),
            is_gated_community   = int(display_data.get("is_gated_community", 0) or 0),
            distance_to_school_km   = display_data.get("distance_to_school_km"),
            distance_to_hospital_km = display_data.get("distance_to_hospital_km"),
            crime_index          = display_data.get("crime_index"),
            road_quality         = display_data.get("road_quality"),
            predicted_price      = prediction,
        )
        db.session.add(v)
        db.session.commit()

        return render_template(
            "sale_result.html",
            price=f"{prediction:,}",
            data=display_data,
            valuation_id=v.id,
        )

    return render_template("sale.html", current_year=datetime.now().year)


@app.route("/download_sale_report")
@login_required
@role_required("owner")
def download_sale_report():
    price    = request.args.get("price")
    pdf_bytes = build_pdf_report(
        "AI Property Valuation Report", "Sale Valuation",
        "Estimated Sale Price", price, request.args,
    )
    response = make_response(pdf_bytes)
    response.headers["Content-Type"]        = "application/pdf"
    response.headers["Content-Disposition"] = "attachment; filename=sale_report.pdf"
    return response


# Download from saved valuation record
@app.route("/valuations/<int:valuation_id>/download")
@login_required
@role_required("owner")
def download_valuation_report(valuation_id):
    v = Valuation.query.get_or_404(valuation_id)
    if v.user_id != current_user.id:
        from flask import abort; abort(403)

    args = {
        "neighborhood": v.neighborhood, "town": v.town,
        "property_type": v.property_type, "bedrooms": v.bedrooms,
        "bathrooms": v.bathrooms, "floor_size_sqm": v.floor_size_sqm,
        "year_built": v.year_built, "condition": v.condition,
        "furnishing": v.furnishing,
        "distance_to_cbd_km": v.distance_to_cbd_km,
        "distance_to_school_km": v.distance_to_school_km,
        "distance_to_hospital_km": v.distance_to_hospital_km,
        "crime_index": v.crime_index, "road_quality": v.road_quality,
    }
    label      = "Estimated Sale Price" if v.valuation_type == "sale" else "Estimated Monthly Rent"
    subtitle   = "Sale Valuation" if v.valuation_type == "sale" else "Rental Valuation"
    pdf_bytes  = build_pdf_report(
        "AI Property Valuation Report", subtitle, label,
        f"{v.predicted_price:,}", args,
    )
    response = make_response(pdf_bytes)
    response.headers["Content-Type"]        = "application/pdf"
    response.headers["Content-Disposition"] = f"attachment; filename={v.valuation_type}_report_{v.id}.pdf"
    return response


# ── Rental valuation ──────────────────────────────────────────────────────────

@app.route("/rental", methods=["GET", "POST"])
@login_required
@role_required("owner")
def rental():
    if request.method == "POST":
        data     = request.form.to_dict()
        selected = data["neighborhood"]

        loc = get_location_row(rental_location_lookup, selected)
        if loc is None:
            flash("Invalid location selected.")
            return redirect(url_for("rental"))

        data["distance_to_cbd_km"]      = loc["distance_to_cbd_km"]
        data["location_tier"]           = loc["location_tier"]
        data["distance_to_school_km"]   = loc.get("distance_to_school_km")
        data["distance_to_hospital_km"] = loc.get("distance_to_hospital_km")
        data["crime_index"]             = loc.get("crime_index")
        data["road_quality"]            = loc.get("road_quality")

        display_data = data.copy()

        df = prepare_rental_input(data, rental_encoders)
        df = df.reindex(columns=rental_model.feature_names_in_, fill_value=0)

        prediction = round(rental_model.predict(df)[0])

        v = Valuation(
            user_id              = current_user.id,
            label                = data.get("label") or None,
            valuation_type       = "rental",
            town                 = display_data["town"],
            neighborhood         = display_data["neighborhood"],
            distance_to_cbd_km   = display_data.get("distance_to_cbd_km"),
            location_tier        = display_data.get("location_tier"),
            property_type        = display_data["property_type"],
            bedrooms             = int(display_data.get("bedrooms", 0) or 0),
            bathrooms            = int(display_data.get("bathrooms", 0) or 0),
            floor_size_sqm       = int(display_data.get("floor_size_sqm", 0) or 0),
            year_built           = int(display_data.get("year_built", 2000) or 2000),
            parking_spaces       = int(display_data.get("parking_spaces", 0) or 0),
            floor_number         = int(display_data.get("floor_number", 0) or 0),
            condition            = display_data.get("condition"),
            furnishing           = display_data.get("furnishing"),
            has_swimming_pool    = int(display_data.get("has_swimming_pool", 0) or 0),
            has_gym              = int(display_data.get("has_gym", 0) or 0),
            has_borehole         = int(display_data.get("has_borehole", 0) or 0),
            has_backup_generator = int(display_data.get("has_backup_generator", 0) or 0),
            has_security         = int(display_data.get("has_security", 0) or 0),
            has_garden           = int(display_data.get("has_garden", 0) or 0),
            is_gated_community   = int(display_data.get("is_gated_community", 0) or 0),
            distance_to_school_km   = display_data.get("distance_to_school_km"),
            distance_to_hospital_km = display_data.get("distance_to_hospital_km"),
            crime_index          = display_data.get("crime_index"),
            road_quality         = display_data.get("road_quality"),
            predicted_price      = prediction,
        )
        db.session.add(v)
        db.session.commit()

        return render_template(
            "rental_result.html",
            price=f"{prediction:,}",
            data=display_data,
            valuation_id=v.id,
        )

    return render_template("rental.html", current_year=datetime.now().year)


@app.route("/download_rental_report")
@login_required
@role_required("owner")
def download_rental_report():
    price    = request.args.get("price")
    pdf_bytes = build_pdf_report(
        "AI Property Valuation Report", "Rental Valuation",
        "Estimated Monthly Rent", price, request.args,
    )
    response = make_response(pdf_bytes)
    response.headers["Content-Type"]        = "application/pdf"
    response.headers["Content-Disposition"] = "attachment; filename=rental_report.pdf"
    return response


# ── Valuations CRUD ───────────────────────────────────────────────────────────

@app.route("/valuations")
@login_required
@role_required("owner")
def valuations_history():
    active_type = request.args.get("type", "sale")

    valuations = (Valuation.query
                  .filter_by(user_id=current_user.id, valuation_type=active_type)
                  .order_by(Valuation.created_at.desc())
                  .all())

    sale_count   = Valuation.query.filter_by(user_id=current_user.id, valuation_type="sale").count()
    rental_count = Valuation.query.filter_by(user_id=current_user.id, valuation_type="rental").count()

    return render_template(
        "valuations_history.html",
        valuations=valuations,
        active_type=active_type,
        sale_count=sale_count,
        rental_count=rental_count,
    )


@app.route("/valuations/<int:valuation_id>")
@login_required
@role_required("owner")
def view_valuation(valuation_id):
    v = Valuation.query.get_or_404(valuation_id)
    if v.user_id != current_user.id:
        from flask import abort; abort(403)
    return render_template("valuation_detail.html", v=v)


@app.route("/valuations/<int:valuation_id>/edit", methods=["GET", "POST"])
@login_required
@role_required("owner")
def edit_valuation(valuation_id):
    v = Valuation.query.get_or_404(valuation_id)
    if v.user_id != current_user.id:
        from flask import abort; abort(403)

    if request.method == "POST":
        data     = request.form.to_dict()
        selected = data["neighborhood"]

        lookup = sale_location_lookup if v.valuation_type == "sale" else rental_location_lookup
        loc    = get_location_row(lookup, selected)

        if loc is None:
            flash("Invalid location selected.")
            return redirect(url_for("edit_valuation", valuation_id=v.id))

        data["distance_to_cbd_km"]      = loc["distance_to_cbd_km"]
        data["location_tier"]           = loc["location_tier"]
        data["distance_to_school_km"]   = loc.get("distance_to_school_km")
        data["distance_to_hospital_km"] = loc.get("distance_to_hospital_km")
        data["crime_index"]             = loc.get("crime_index")
        data["road_quality"]            = loc.get("road_quality")

        display_data = data.copy()

        if v.valuation_type == "sale":
            df = prepare_sale_input(data, sale_encoders)
            df = df.reindex(columns=sale_model.feature_names_in_, fill_value=0)
            prediction = round(sale_model.predict(df)[0])
        else:
            df = prepare_rental_input(data, rental_encoders)
            df = df.reindex(columns=rental_model.feature_names_in_, fill_value=0)
            prediction = round(rental_model.predict(df)[0])

        # Update record
        v.label                = display_data.get("label") or None
        v.town                 = display_data["town"]
        v.neighborhood         = display_data["neighborhood"]
        v.distance_to_cbd_km   = display_data.get("distance_to_cbd_km")
        v.location_tier        = display_data.get("location_tier")
        v.property_type        = display_data["property_type"]
        v.bedrooms             = int(display_data.get("bedrooms", 0) or 0)
        v.bathrooms            = int(display_data.get("bathrooms", 0) or 0)
        v.floor_size_sqm       = int(display_data.get("floor_size_sqm", 0) or 0)
        v.year_built           = int(display_data.get("year_built", 2000) or 2000)
        v.parking_spaces       = int(display_data.get("parking_spaces", 0) or 0)
        v.floor_number         = int(display_data.get("floor_number", 0) or 0)
        v.condition            = display_data.get("condition")
        v.furnishing           = display_data.get("furnishing")
        v.has_swimming_pool    = int(display_data.get("has_swimming_pool", 0) or 0)
        v.has_gym              = int(display_data.get("has_gym", 0) or 0)
        v.has_borehole         = int(display_data.get("has_borehole", 0) or 0)
        v.has_backup_generator = int(display_data.get("has_backup_generator", 0) or 0)
        v.has_security         = int(display_data.get("has_security", 0) or 0)
        v.has_garden           = int(display_data.get("has_garden", 0) or 0)
        v.is_gated_community   = int(display_data.get("is_gated_community", 0) or 0)
        v.distance_to_school_km   = display_data.get("distance_to_school_km")
        v.distance_to_hospital_km = display_data.get("distance_to_hospital_km")
        v.crime_index          = display_data.get("crime_index")
        v.road_quality         = display_data.get("road_quality")
        v.predicted_price      = prediction
        v.updated_at           = datetime.utcnow()

        db.session.commit()

        flash("Valuation updated successfully.")
        return redirect(url_for("view_valuation", valuation_id=v.id))

    return render_template("edit_valuation.html", v=v)


@app.route("/valuations/<int:valuation_id>/delete", methods=["POST"])
@login_required
@role_required("owner")
def delete_valuation(valuation_id):
    v = Valuation.query.get_or_404(valuation_id)
    if v.user_id != current_user.id:
        from flask import abort; abort(403)
    vtype = v.valuation_type
    db.session.delete(v)
    db.session.commit()
    flash("Valuation deleted.")
    return redirect(url_for("valuations_history", type=vtype))


# ── Insurance assessment ──────────────────────────────────────────────────────

@app.route("/insurance-assessment", methods=["GET", "POST"])
@login_required
@role_required("insurance")
def insurance_assessment():
    if request.method == "POST":
        data     = request.form.to_dict()
        selected = data["neighborhood"]

        loc = get_location_row(sale_location_lookup, selected)
        if loc is None:
            flash("Invalid location selected.")
            return redirect(url_for("insurance_assessment"))

        data["distance_to_cbd_km"]      = loc["distance_to_cbd_km"]
        data["location_tier"]           = loc["location_tier"]
        data["distance_to_school_km"]   = loc.get("distance_to_school_km")
        data["distance_to_hospital_km"] = loc.get("distance_to_hospital_km")
        data["crime_index"]             = loc.get("crime_index")
        data["road_quality"]            = loc.get("road_quality")

        display_data = data.copy()

        df = prepare_sale_input(data, sale_encoders)
        df = df.reindex(columns=sale_model.feature_names_in_, fill_value=0)
        prediction = round(sale_model.predict(df)[0])

        risk_tier, risk_factors, coverage_low, coverage_high = assess_risk(
            display_data, prediction
        )

        a = InsuranceAssessment(
            user_id              = current_user.id,
            label                = display_data.get("label") or None,
            town                 = display_data["town"],
            neighborhood         = display_data["neighborhood"],
            distance_to_cbd_km   = display_data.get("distance_to_cbd_km"),
            location_tier        = display_data.get("location_tier"),
            property_type        = display_data["property_type"],
            bedrooms             = int(display_data.get("bedrooms", 0) or 0),
            bathrooms            = int(display_data.get("bathrooms", 0) or 0),
            floor_size_sqm       = int(display_data.get("floor_size_sqm", 0) or 0),
            year_built           = int(display_data.get("year_built", 2000) or 2000),
            parking_spaces       = int(display_data.get("parking_spaces", 0) or 0),
            floor_number         = int(display_data.get("floor_number", 0) or 0),
            condition            = display_data.get("condition"),
            furnishing           = display_data.get("furnishing"),
            has_swimming_pool    = int(display_data.get("has_swimming_pool", 0) or 0),
            has_gym              = int(display_data.get("has_gym", 0) or 0),
            has_borehole         = int(display_data.get("has_borehole", 0) or 0),
            has_backup_generator = int(display_data.get("has_backup_generator", 0) or 0),
            has_security         = int(display_data.get("has_security", 0) or 0),
            has_garden           = int(display_data.get("has_garden", 0) or 0),
            is_gated_community   = int(display_data.get("is_gated_community", 0) or 0),
            distance_to_school_km   = display_data.get("distance_to_school_km"),
            distance_to_hospital_km = display_data.get("distance_to_hospital_km"),
            crime_index          = display_data.get("crime_index"),
            road_quality         = display_data.get("road_quality"),
            predicted_value      = prediction,
            risk_tier            = risk_tier,
            coverage_low         = int(coverage_low),
            coverage_high        = int(coverage_high),
        )
        db.session.add(a)
        db.session.commit()

        return render_template(
            "insurance_result.html",
            price=f"{prediction:,}",
            risk_tier=risk_tier,
            risk_factors=risk_factors,
            coverage_low=f"{int(coverage_low):,}",
            coverage_high=f"{int(coverage_high):,}",
            data=display_data,
            assessment_id=a.id,
        )

    return render_template("insurance_assessment.html", current_year=datetime.now().year)


# Insurance assessments CRUD

@app.route("/insurance/assessments")
@login_required
@role_required("insurance")
def insurance_assessments_history():
    assessments = (InsuranceAssessment.query
                   .filter_by(user_id=current_user.id)
                   .order_by(InsuranceAssessment.created_at.desc())
                   .all())
    return render_template(
        "insurance_assessments_history.html",
        assessments=assessments,
        total=len(assessments),
    )


@app.route("/insurance/assessments/<int:assessment_id>")
@login_required
@role_required("insurance")
def view_insurance_assessment(assessment_id):
    a = InsuranceAssessment.query.get_or_404(assessment_id)
    if a.user_id != current_user.id:
        from flask import abort; abort(403)
    return render_template("insurance_result_detail.html", a=a)


@app.route("/insurance/assessments/<int:assessment_id>/delete", methods=["POST"])
@login_required
@role_required("insurance")
def delete_insurance_assessment(assessment_id):
    a = InsuranceAssessment.query.get_or_404(assessment_id)
    if a.user_id != current_user.id:
        from flask import abort; abort(403)
    db.session.delete(a)
    db.session.commit()
    flash("Assessment deleted.")
    return redirect(url_for("insurance_assessments_history"))


@app.route("/insurance/assessments/<int:assessment_id>/download")
@login_required
@role_required("insurance")
def download_insurance_assessment_report(assessment_id):
    a = InsuranceAssessment.query.get_or_404(assessment_id)
    if a.user_id != current_user.id:
        from flask import abort; abort(403)

    args = {
        "neighborhood": a.neighborhood, "town": a.town,
        "property_type": a.property_type, "bedrooms": a.bedrooms,
        "bathrooms": a.bathrooms, "floor_size_sqm": a.floor_size_sqm,
        "year_built": a.year_built, "condition": a.condition,
        "furnishing": a.furnishing,
        "distance_to_cbd_km": a.distance_to_cbd_km,
        "distance_to_school_km": a.distance_to_school_km,
        "distance_to_hospital_km": a.distance_to_hospital_km,
        "crime_index": a.crime_index, "road_quality": a.road_quality,
    }
    pdf_bytes = build_pdf_report(
        "PropAI Insurance Risk Assessment Report",
        "Insurance Risk Assessment",
        "Estimated Property Value",
        f"{a.predicted_value:,}",
        args,
        extra_sections=[("Risk Assessment", [
            ("Risk Tier",              a.risk_tier),
            ("Suggested Coverage Range", f"KES {a.coverage_low:,} – KES {a.coverage_high:,}"),
        ])],
    )
    response = make_response(pdf_bytes)
    response.headers["Content-Type"]        = "application/pdf"
    response.headers["Content-Disposition"] = f"attachment; filename=insurance_report_{a.id}.pdf"
    return response


@app.route("/download_insurance_report")
@login_required
@role_required("insurance")
def download_insurance_report():
    price        = request.args.get("price")
    risk_tier    = request.args.get("risk_tier")
    coverage_low = request.args.get("coverage_low")
    coverage_high= request.args.get("coverage_high")

    pdf_bytes = build_pdf_report(
        "PropAI Insurance Risk Assessment Report",
        "Insurance Risk Assessment",
        "Estimated Property Value", price,
        request.args,
        extra_sections=[("Risk Assessment", [
            ("Risk Tier",              risk_tier or "-"),
            ("Suggested Coverage Range", f"KES {coverage_low} – KES {coverage_high}"),
        ])],
    )
    response = make_response(pdf_bytes)
    response.headers["Content-Type"]        = "application/pdf"
    response.headers["Content-Disposition"] = "attachment; filename=insurance_risk_assessment.pdf"
    return response


# ── Admin routes ──────────────────────────────────────────────────────────────

@app.route("/admin")
@login_required
@role_required("admin")
def admin_redirect():
    return redirect(url_for("admin_dashboard_grid"))


@app.route("/admin/dashboard")
@login_required
@role_required("admin")
def admin_dashboard_grid():
    return render_template("admin_dashboard.html")


@app.route("/admin/users")
@login_required
@role_required("admin")
def user_management():
    users = User.query.order_by(User.id).all()
    return render_template("user_management.html", users=users)


@app.route("/admin/users/<int:user_id>/role", methods=["POST"])
@login_required
@role_required("admin")
def update_user_role(user_id):
    user = User.query.get(user_id)
    if not user:
        flash("User not found.")
        return redirect(url_for("user_management"))
    if user.id == current_user.id:
        flash("You cannot change your own role.")
        return redirect(url_for("user_management"))
    new_role = request.form.get("role")
    if new_role not in ("owner", "insurance", "admin"):
        flash("Invalid role.")
        return redirect(url_for("user_management"))
    user.role = new_role
    db.session.commit()
    flash(f"Updated {user.email} to role '{new_role}'.")
    return redirect(url_for("user_management"))


@app.route("/admin/performance")
@login_required
@role_required("admin")
def performance_monitoring():
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.ensemble import RandomForestRegressor

    TARGET       = "price_ksh"
    DROP_COLS    = ["property_id", "listing_type", "land_size_acres"]
    TEXT_COLS    = ["property_type","neighborhood","town","condition",
                    "furnishing","location_tier","road_quality"]

    def compute(df_subset):
        df = df_subset.copy().dropna(subset=[TARGET])
        df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
        for col in TEXT_COLS:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
        X = df.drop(columns=[TARGET])
        y = df[TARGET]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        m = RandomForestRegressor(n_estimators=200, random_state=42)
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        try:
            mae  = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2   = r2_score(y_test, preds)
            return {"mae": f"{mae:,.0f}", "rmse": f"{rmse:,.0f}", "r2": f"{r2:.3f}", "n_rows": len(X_test)}
        except Exception:
            return {"mae": "N/A", "rmse": "N/A", "r2": "N/A", "n_rows": 0}

    sale_metrics   = compute(sale_location_df)
    rental_metrics = compute(rental_location_df)

    usage = {
        "total_users":     User.query.count(),
        "owner_count":     User.query.filter_by(role="owner").count(),
        "insurance_count": User.query.filter_by(role="insurance").count(),
        "admin_count":     User.query.filter_by(role="admin").count(),
        "total_valuations":    Valuation.query.count(),
        "total_assessments":   InsuranceAssessment.query.count(),
    }

    return render_template("performance_monitoring.html",
                           sale_metrics=sale_metrics,
                           rental_metrics=rental_metrics,
                           usage=usage)


@app.route("/admin/dataset")
@login_required
@role_required("admin")
def dataset_management():
    dataset_type = request.args.get("dataset", "sale")
    search_query = request.args.get("q", "").strip()
    page         = max(1, request.args.get("page", 1, type=int))

    df = sale_location_df if dataset_type == "sale" else rental_location_df

    if search_query:
        mask = (
            df["neighborhood"].astype(str).str.contains(search_query, case=False, na=False) |
            df["town"].astype(str).str.contains(search_query, case=False, na=False)
        )
        filtered = df[mask]
    else:
        filtered = df

    total_rows  = len(filtered)
    per_page    = 50
    total_pages = max(1, (total_rows + per_page - 1) // per_page)
    page        = min(page, total_pages)
    start       = (page - 1) * per_page
    end         = start + per_page

    return render_template(
        "dataset_management.html",
        dataset_type=dataset_type,
        rows=filtered.iloc[start:end].to_dict(orient="records"),
        columns=list(df.columns),
        total_rows=total_rows,
        search_query=search_query,
        page=page,
        total_pages=total_pages,
        start_row=start + 1 if total_rows > 0 else 0,
        end_row=min(end, total_rows),
    )


@app.route("/get_neighborhood_details_admin/<property_type>/<town>")
@login_required
@role_required("admin")
def get_neighborhood_details_admin(property_type, town):
    df     = sale_location_df if property_type == "sale" else rental_location_df
    lookup = sale_location_lookup if property_type == "sale" else rental_location_lookup
    nh     = df[df["town"] == town]["neighborhood"].unique()
    matches= lookup[lookup["neighborhood"].isin(nh)]
    results= [
        {
            "neighborhood":            row["neighborhood"],
            "distance_to_cbd_km":      row.get("distance_to_cbd_km"),
            "location_tier":           row.get("location_tier"),
            "distance_to_school_km":   row.get("distance_to_school_km"),
            "distance_to_hospital_km": row.get("distance_to_hospital_km"),
            "crime_index":             row.get("crime_index"),
            "road_quality":            row.get("road_quality"),
        }
        for _, row in matches.iterrows()
    ]
    return {"neighborhoods": results}


@app.route("/admin/dataset/add", methods=["POST"])
@login_required
@role_required("admin")
def add_property_row():
    dataset_type = request.form.get("dataset_type", "sale")
    csv_path     = DATASET_PATH
    town         = request.form.get("town")
    neighborhood = request.form.get("neighborhood")
    distance     = request.form.get("distance_to_cbd_km")
    tier         = request.form.get("location_tier")

    if not distance or not tier:
        flash("Location details are missing. Pick an existing neighborhood from the dropdown.")
        return redirect(url_for("dataset_management", dataset=dataset_type))

    if not request.form.get("target_price"):
        flash("Price is required.")
        return redirect(url_for("dataset_management", dataset=dataset_type))

    new_row = {
        "listing_type":          "For Sale" if dataset_type == "sale" else "For Rent",
        "town":                  town,
        "neighborhood":          neighborhood,
        "distance_to_cbd_km":    distance,
        "location_tier":         tier,
        "property_type":         request.form.get("property_type"),
        "bedrooms":              request.form.get("bedrooms"),
        "bathrooms":             request.form.get("bathrooms"),
        "floor_size_sqm":        request.form.get("floor_size_sqm"),
        "year_built":            request.form.get("year_built"),
        "parking_spaces":        request.form.get("parking_spaces"),
        "floor_number":          request.form.get("floor_number"),
        "condition":             request.form.get("condition"),
        "furnishing":            request.form.get("furnishing"),
        "has_swimming_pool":     int(request.form.get("has_swimming_pool", 0) or 0),
        "has_gym":               int(request.form.get("has_gym", 0) or 0),
        "has_borehole":          int(request.form.get("has_borehole", 0) or 0),
        "has_backup_generator":  int(request.form.get("has_backup_generator", 0) or 0),
        "has_security":          int(request.form.get("has_security", 0) or 0),
        "has_garden":            int(request.form.get("has_garden", 0) or 0),
        "is_gated_community":    int(request.form.get("is_gated_community", 0) or 0),
        "price_ksh":             request.form.get("target_price"),
        "distance_to_school_km":   request.form.get("distance_to_school_km"),
        "distance_to_hospital_km": request.form.get("distance_to_hospital_km"),
        "crime_index":           request.form.get("crime_index"),
        "road_quality":          request.form.get("road_quality"),
    }

    try:
        existing = pd.read_csv(csv_path)
        if "property_id" in existing.columns:
            ids = pd.to_numeric(existing["property_id"], errors="coerce")
            new_row["property_id"] = int(ids.max() + 1) if ids.notna().any() else 1
        filtered = {k: v for k, v in new_row.items() if k in existing.columns}
        updated  = pd.concat([existing, pd.DataFrame([filtered])], ignore_index=True)
        updated.to_csv(csv_path, index=False)
        flash(f"Added listing in {neighborhood}, {town}. Retrain models to include it in predictions.")
    except Exception as e:
        flash(f"Failed to add listing: {e}")

    return redirect(url_for("dataset_management", dataset=dataset_type))


@app.route("/admin/dataset/add_neighborhood", methods=["POST"])
@login_required
@role_required("admin")
def add_neighborhood():
    dataset_type = request.form.get("dataset_type", "sale")
    neighborhood = request.form.get("neighborhood", "").strip()
    town         = request.form.get("town")
    distance_to_cbd_km   = request.form.get("distance_to_cbd_km")
    location_tier        = request.form.get("location_tier")
    distance_to_school_km   = request.form.get("distance_to_school_km")
    distance_to_hospital_km = request.form.get("distance_to_hospital_km")
    crime_index  = request.form.get("crime_index")
    road_quality = request.form.get("road_quality")

    if not neighborhood or not town or not distance_to_cbd_km:
        flash("Neighborhood name, town and distance to CBD are required.")
        return redirect(url_for("dataset_management", dataset=dataset_type))

    try:
        existing = pd.read_csv(DATASET_PATH)
        
        # Ensure road_quality column is string type
        if "road_quality" in existing.columns:
            existing["road_quality"] = existing["road_quality"].astype(str)

        exists = (
            (existing["town"] == town) &
            (existing["neighborhood"].str.lower() == neighborhood.lower())
        ).any()

        if exists:
            flash(f"'{neighborhood}' already exists in {town}. Use the listing form to add properties there.")
            return redirect(url_for("dataset_management", dataset=dataset_type))

        # Build a minimal placeholder row
        placeholder = {
            "listing_type":          "For Sale" if dataset_type == "sale" else "For Rent",
            "town":                  town,
            "neighborhood":          neighborhood,
            "distance_to_cbd_km":    float(distance_to_cbd_km),
            "location_tier":         location_tier or "mid",
            "property_type":         "Apartment",
            "bedrooms":              0,
            "bathrooms":             0,
            "floor_size_sqm":        0,
            "year_built":            2020,
            "parking_spaces":        0,
            "floor_number":          0,
            "condition":             "Good",
            "furnishing":            "Unfurnished",
            "has_swimming_pool":     0,
            "has_gym":               0,
            "has_borehole":          0,
            "has_backup_generator":  0,
            "has_security":          0,
            "has_garden":            0,
            "is_gated_community":    0,
            "price_ksh":             None,
            "distance_to_school_km":   float(distance_to_school_km) if distance_to_school_km and distance_to_school_km.strip() else None,
            "distance_to_hospital_km": float(distance_to_hospital_km) if distance_to_hospital_km and distance_to_hospital_km.strip() else None,
            "crime_index":           float(crime_index) if crime_index and crime_index.strip() else None,
            "road_quality":          road_quality or "Paved",  # This is a string
        }

        if "property_id" in existing.columns:
            ids = pd.to_numeric(existing["property_id"], errors="coerce")
            placeholder["property_id"] = int(ids.max() + 1) if ids.notna().any() else 1

        filtered = {k: v for k, v in placeholder.items() if k in existing.columns}
        
        # Convert the new row to DataFrame with proper types
        new_row_df = pd.DataFrame([filtered])
        
        # Ensure road_quality is string in the new row
        if "road_quality" in new_row_df.columns:
            new_row_df["road_quality"] = new_row_df["road_quality"].astype(str)
        
        updated = pd.concat([existing, new_row_df], ignore_index=True)
        updated.to_csv(DATASET_PATH, index=False)

        flash(
            f"'{neighborhood}' in {town} has been added. It will now appear in the "
            f"neighborhood dropdowns. Add actual property listings below, then "
            f"retrain the models to include this area in predictions."
        )

    except Exception as e:
        flash(f"Failed to add neighborhood: {e}")

    return redirect(url_for("dataset_management", dataset=dataset_type))

@app.route("/admin/neighborhood-factors")
@login_required
@role_required("admin")
def neighborhood_factors():
    search = request.args.get("q", "").strip()
    page   = max(1, request.args.get("page", 1, type=int))

    # RELOAD the CSV to get the latest data
    # This is the key fix - always read fresh data
    full_df_fresh = pd.read_csv(DATASET_PATH)
    
    # Build neighborhood-level summary (unique neighborhood/town pairs + factors)
    factor_cols = ["neighborhood", "town", "distance_to_cbd_km", "location_tier",
                   "distance_to_school_km", "distance_to_hospital_km",
                   "crime_index", "road_quality"]
    available   = [c for c in factor_cols if c in full_df_fresh.columns]
    nh_df       = full_df_fresh[available].drop_duplicates(subset=["neighborhood", "town"])
    
    # Ensure road_quality is string type for display
    if "road_quality" in nh_df.columns:
        nh_df["road_quality"] = nh_df["road_quality"].astype(str)

    if search:
        mask  = (
            nh_df["neighborhood"].str.contains(search, case=False, na=False) |
            nh_df["town"].str.contains(search, case=False, na=False)
        )
        nh_df = nh_df[mask]

    total      = len(nh_df)
    per_page   = 20
    total_pages= max(1, (total + per_page - 1) // per_page)
    page       = min(page, total_pages)
    start      = (page - 1) * per_page
    rows       = nh_df.iloc[start:start+per_page].to_dict(orient="records")

    return render_template(
        "neighborhood_factors.html",
        rows=rows, total=total,
        search_query=search,
        page=page, total_pages=total_pages,
        start_row=start+1 if total > 0 else 0,
        end_row=min(start+per_page, total),
    )


@app.route("/admin/neighborhood-factors/update", methods=["POST"])
@login_required
@role_required("admin")
def update_neighborhood_factor():
    neighborhood = request.form.get("neighborhood")
    town         = request.form.get("town")

    try:
        df = pd.read_csv(DATASET_PATH)
        
        # Ensure road_quality column is string type
        if "road_quality" in df.columns:
            df["road_quality"] = df["road_quality"].astype(str)
        
        mask = (df["neighborhood"] == neighborhood) & (df["town"] == town)

        if not mask.any():
            flash(f"Neighborhood '{neighborhood}' in {town} not found.")
            return redirect(url_for("neighborhood_factors"))

        # Update distance_to_school_km (float)
        if "distance_to_school_km" in df.columns:
            try:
                val = request.form.get("distance_to_school_km")
                if val and val.strip():
                    df.loc[mask, "distance_to_school_km"] = float(val)
            except (ValueError, TypeError):
                flash("Invalid value for distance_to_school_km. Must be a number.")
                return redirect(url_for("neighborhood_factors"))
                
        # Update distance_to_hospital_km (float)
        if "distance_to_hospital_km" in df.columns:
            try:
                val = request.form.get("distance_to_hospital_km")
                if val and val.strip():
                    df.loc[mask, "distance_to_hospital_km"] = float(val)
            except (ValueError, TypeError):
                flash("Invalid value for distance_to_hospital_km. Must be a number.")
                return redirect(url_for("neighborhood_factors"))
                
        # Update crime_index (float)
        if "crime_index" in df.columns:
            try:
                val = request.form.get("crime_index")
                if val and val.strip():
                    df.loc[mask, "crime_index"] = float(val)
            except (ValueError, TypeError):
                flash("Invalid value for crime_index. Must be a number.")
                return redirect(url_for("neighborhood_factors"))
                
        # Update road_quality (string)
        if "road_quality" in df.columns:
            road_quality = request.form.get("road_quality")
            if road_quality in ['Paved', 'Mixed', 'Murram']:
                # Convert the entire column to string first
                df["road_quality"] = df["road_quality"].astype(str)
                df.loc[mask, "road_quality"] = road_quality
                print(f"Updated road_quality for {neighborhood} to: {road_quality}")  # Debug log
            else:
                flash("Invalid road quality value. Must be Paved, Mixed, or Murram.")
                return redirect(url_for("neighborhood_factors"))

        # Save the CSV
        df.to_csv(DATASET_PATH, index=False)
        flash(f"Updated factors for {neighborhood}, {town}. Retrain models to apply changes.")

    except Exception as e:
        flash(f"Update failed: {e}")

    return redirect(url_for("neighborhood_factors"))

@app.route("/admin/retrain")
@login_required
@role_required("admin")
def model_retraining():
    return render_template(
        "model_retraining.html",
        sale_row_count=len(sale_location_df),
        rental_row_count=len(rental_location_df),
    )


@app.route("/admin/retrain/<model_type>", methods=["POST"])
@login_required
@role_required("admin")
def retrain_model(model_type):
    import shutil
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    if model_type not in ("sale", "rental"):
        flash("Invalid model type.")
        return redirect(url_for("model_retraining"))

    listing_type = "For Sale" if model_type == "sale" else "For Rent"
    model_path   = f"models/{model_type}_model.pkl"
    enc_path     = f"models/{model_type}_encoders.pkl"
    TARGET       = "price_ksh"
    DROP_COLS    = ["property_id", "listing_type", "land_size_acres"]
    TEXT_COLS    = ["property_type","neighborhood","town","condition",
                    "furnishing","location_tier","road_quality"]

    try:
        df = pd.read_csv(DATASET_PATH)
        df = df[df["listing_type"] == listing_type].copy()

        if TARGET not in df.columns:
            flash(f"Target column '{TARGET}' not found. Aborted.")
            return redirect(url_for("model_retraining"))

        df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
        df = df.dropna(subset=[TARGET])

        encoders = {}
        for col in TEXT_COLS:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le

        X = df.drop(columns=[TARGET])
        y = df[TARGET]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mae   = mean_absolute_error(y_test, preds)
        rmse  = np.sqrt(mean_squared_error(y_test, preds))
        r2    = r2_score(y_test, preds)

        if os.path.exists(model_path):
            shutil.copy(model_path, model_path.replace(".pkl", "_backup.pkl"))
        if os.path.exists(enc_path):
            shutil.copy(enc_path, enc_path.replace(".pkl", "_backup.pkl"))

        with open(model_path, "wb") as f: pickle.dump(model, f)
        with open(enc_path,   "wb") as f: pickle.dump(encoders, f)

        flash(f"{model_type.capitalize()} model retrained on {len(X_train)} rows. "
              f"MAE: KES {mae:,.0f} | RMSE: KES {rmse:,.0f} | R²: {r2:.3f}. "
              f"Restart the app to load the new model.")

    except Exception as e:
        flash(f"Retraining failed: {e}")

    return redirect(url_for("model_retraining"))


# ── Error handlers ────────────────────────────────────────────────────────────

@app.errorhandler(403)
def forbidden(e):
    return render_template("403.html"), 403


if __name__ == "__main__":
    app.run(debug=True)