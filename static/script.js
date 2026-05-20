document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("loan-application-form");
    const submitBtn = document.getElementById("submit-btn");
    const btnText = submitBtn.querySelector(".btn-text");
    const btnLoader = submitBtn.querySelector(".btn-loader");
    
    const resultsCard = document.getElementById("results-card");
    const placeholderView = document.getElementById("placeholder-view");
    const resultsView = document.getElementById("results-view");
    
    const decisionBadge = document.getElementById("decision-badge");
    const decisionIcon = document.getElementById("decision-icon");
    const decisionText = document.getElementById("decision-text");
    const statusSummary = document.getElementById("status-summary");
    
    const probabilityRing = document.getElementById("probability-ring");
    const probabilityVal = document.getElementById("probability-val");
    
    const amountBox = document.getElementById("amount-box");
    const recommendedAmountVal = document.getElementById("recommended-amount-val");
    
    const tableCibil = document.getElementById("table-cibil");
    const tableIncomeRatio = document.getElementById("table-income-ratio");
    const tableTotalAssets = document.getElementById("table-total-assets");
    
    // CIBIL Slider Logic
    const cibilSlider = document.getElementById("cibil_score");
    const cibilValDisplay = document.getElementById("cibil-val-display");
    
    function updateCibilBadge(score) {
        cibilValDisplay.textContent = score;
        cibilValDisplay.className = "badge"; // Reset classes
        
        if (score >= 800) {
            cibilValDisplay.classList.add("score-excellent");
            cibilValDisplay.textContent = `${score} (Excellent)`;
        } else if (score >= 700) {
            cibilValDisplay.classList.add("score-good");
            cibilValDisplay.textContent = `${score} (Good)`;
        } else if (score >= 550) {
            cibilValDisplay.classList.add("score-fair");
            cibilValDisplay.textContent = `${score} (Fair)`;
        } else {
            cibilValDisplay.classList.add("score-poor");
            cibilValDisplay.textContent = `${score} (Poor)`;
        }
    }
    
    cibilSlider.addEventListener("input", (e) => {
        updateCibilBadge(parseInt(e.target.value));
    });
    
    // Initialize CIBIL Badge
    updateCibilBadge(parseInt(cibilSlider.value));

    // Form Submission
    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        
        // Disable button & show loader
        submitBtn.disabled = true;
        btnText.classList.add("hidden");
        btnLoader.classList.remove("hidden");
        
        // Gather data
        const formData = new FormData(form);
        const data = {
            no_of_dependents: parseInt(formData.get("no_of_dependents")),
            education: formData.get("education"),
            self_employed: formData.get("self_employed"),
            income_annum: parseInt(formData.get("income_annum")),
            loan_term: parseInt(formData.get("loan_term")),
            cibil_score: parseInt(formData.get("cibil_score")),
            residential_assets_value: parseInt(formData.get("residential_assets_value")),
            commercial_assets_value: parseInt(formData.get("commercial_assets_value")),
            luxury_assets_value: parseInt(formData.get("luxury_assets_value")),
            bank_asset_value: parseInt(formData.get("bank_asset_value")),
        };
        
        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(data)
            });
            
            if (!response.ok) {
                throw new Error("Server error occurred during prediction.");
            }
            
            const result = await response.json();
            
            if (result.status === "success") {
                renderResults(result.data, data);
            } else {
                alert("Evaluation failed: " + result.message);
            }
            
        } catch (error) {
            console.error("Error connecting to API:", error);
            alert("Error connecting to the decision engine. Please make sure the backend is running.");
        } finally {
            // Re-enable button
            submitBtn.disabled = false;
            btnText.classList.remove("hidden");
            btnLoader.classList.add("hidden");
        }
    });
    
    function renderResults(prediction, inputData) {
        // Switch views
        resultsCard.classList.remove("empty-state");
        placeholderView.classList.add("hidden");
        resultsView.classList.remove("hidden");
        
        const approved = prediction.approved;
        const probability = prediction.approval_probability;
        const amount = prediction.recommended_loan_amount;
        
        // 1. Update Decision Badge
        decisionBadge.className = "decision-badge"; // Reset classes
        if (approved) {
            decisionBadge.classList.add("state-approved");
            decisionIcon.className = "fa-solid fa-circle-check";
            decisionText.textContent = "APPROVED";
            statusSummary.textContent = "Based on the predictive evaluation, the applicant meets all risk assessment conditions.";
            
            // Show recommended loan amount
            amountBox.classList.remove("hidden");
            // Format currency in Indian Rupees
            const formattedAmount = Math.round(amount).toLocaleString('en-IN');
            recommendedAmountVal.textContent = formattedAmount;
        } else {
            decisionBadge.classList.add("state-rejected");
            decisionIcon.className = "fa-solid fa-circle-xmark";
            decisionText.textContent = "REJECTED";
            statusSummary.textContent = "Based on the predictive evaluation, the applicant does not meet the credit approval threshold.";
            
            // Hide/Disable recommended loan amount
            amountBox.classList.add("hidden");
        }
        
        // 2. Update SVG Circular Confidence Bar
        const percentage = (probability * 100).toFixed(1);
        probabilityVal.textContent = percentage;
        
        // Circumference of radius 50 is 2 * PI * 50 = 314.16
        const circumference = 314.16;
        const strokeDashoffset = circumference - (probability * circumference);
        probabilityRing.style.strokeDashoffset = strokeDashoffset;
        
        // Update colors depending on approval status
        if (approved) {
            probabilityRing.style.stroke = "var(--success-text)";
        } else {
            probabilityRing.style.stroke = "var(--fail-text)";
        }
        
        // 3. Update Risk Score Summary Table
        // CIBIL Score display
        tableCibil.className = "right-align font-bold";
        if (inputData.cibil_score >= 800) {
            tableCibil.textContent = `${inputData.cibil_score} (Excellent)`;
            tableCibil.style.color = "var(--accent-teal)";
        } else if (inputData.cibil_score >= 700) {
            tableCibil.textContent = `${inputData.cibil_score} (Good)`;
            tableCibil.style.color = "var(--success-text)";
        } else if (inputData.cibil_score >= 550) {
            tableCibil.textContent = `${inputData.cibil_score} (Fair)`;
            tableCibil.style.color = "#f97316";
        } else {
            tableCibil.textContent = `${inputData.cibil_score} (Poor)`;
            tableCibil.style.color = "var(--fail-text)";
        }
        
        // Income Ratio display
        tableIncomeRatio.textContent = `₹${inputData.income_annum.toLocaleString('en-IN')} / yr`;
        
        // Total Assets Calculation
        const totalAssets = inputData.residential_assets_value + 
                            inputData.commercial_assets_value + 
                            inputData.luxury_assets_value + 
                            inputData.bank_asset_value;
        tableTotalAssets.textContent = `₹${totalAssets.toLocaleString('en-IN')}`;
    }
});
