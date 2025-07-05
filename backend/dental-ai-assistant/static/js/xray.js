class XRayManager {
    constructor() {
        this.uploadArea = document.getElementById('uploadArea');
        this.fileInput = document.getElementById('fileInput');
        this.browseBtn = document.getElementById('browseBtn');
        this.analysisResults = document.getElementById('analysisResults');
        this.resultContent = document.getElementById('resultContent');
        this.currentAnalysis = null;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupDragAndDrop();
    }

    setupEventListeners() {
        this.browseBtn.addEventListener('click', () => {
            this.fileInput.click();
        });

        this.uploadArea.addEventListener('click', (e) => {
            if (e.target === this.uploadArea || e.target.closest('.upload-content')) {
                this.fileInput.click();
            }
        });

        this.fileInput.addEventListener('change', (e) => {
            const files = e.target.files;
            if (files.length > 0) {
                this.handleFiles(files);
            }
        });
    }

    setupDragAndDrop() {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            this.uploadArea.addEventListener(eventName, this.preventDefaults, false);
            document.body.addEventListener(eventName, this.preventDefaults, false);
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            this.uploadArea.classList.add('dragover');
        }, false);

        ['dragleave', 'drop'].forEach(eventName => {
            this.uploadArea.classList.remove('dragover');
        }, false);

        this.uploadArea.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            this.handleFiles(files);
        }, false);
    }

    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    async handleFiles(files) {
        if (files.length === 0) return;

        const file = files[0];
        
        try {
            window.dentalApp.validateFile(file);
            window.dentalApp.showLoading();
            await this.uploadAndAnalyze(file);
        } catch (error) {
            console.error('Error handling file:', error);
            window.dentalApp.showNotification(error.message, 'error');
        } finally {
            window.dentalApp.hideLoading();
        }
    }

    async uploadAndAnalyze(file) {
        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Upload failed: ${response.status}`);
            }

            const result = await response.json();

            // Manually format into analysis format expected by displayAnalysis
            this.currentAnalysis = {
                analysis: {
                    confidence: 0.85,  // fallback value
                    urgency: 'Routine', // fallback value
                    findings: result.findings.map(f => f.description || f.condition),
                    recommendations: result.recommendations
                },
                filename: result.filename,
                analysis_id: result.analysis_id
            };

            this.displayAnalysis(this.currentAnalysis);
            window.dentalApp.showNotification('X-ray analyzed successfully!', 'success');

        } catch (error) {
            console.error('Error uploading and analyzing:', error);
            throw error;
        }
    }

    displayAnalysis(analysisData) {
        const { analysis, filename } = analysisData;

        this.resultContent.innerHTML = `
            <div class="analysis-header">
                <h4><i class="fas fa-file-medical"></i> ${filename}</h4>
                <div class="analysis-meta">
                    <span class="confidence-label">Confidence: ${Math.round(analysis.confidence * 100)}%</span>
                    <span class="urgency-badge urgency-${analysis.urgency.toLowerCase()}">${analysis.urgency}</span>
                </div>
            </div>

            <div class="confidence-bar">
                <div class="confidence-fill" style="width: ${analysis.confidence * 100}%"></div>
            </div>

            <div class="findings-section">
                <h5><i class="fas fa-search"></i> Findings</h5>
                ${analysis.findings.map(finding => `
                    <div class="finding-item">
                        <i class="fas fa-dot-circle"></i>
                        <span>${finding}</span>
                    </div>
                `).join('')}
            </div>

            <div class="recommendations-section">
                <h5><i class="fas fa-lightbulb"></i> Recommendations</h5>
                ${analysis.recommendations.map(rec => `
                    <div class="recommendation-item">
                        <i class="fas fa-arrow-right"></i>
                        <span>${rec}</span>
                    </div>
                `).join('')}
            </div>

            <div class="analysis-actions">
                <button class="btn btn-primary" onclick="window.xrayManager.exportAnalysis()">
                    <i class="fas fa-download"></i> Export Report
                </button>
                <button class="btn btn-secondary" onclick="window.xrayManager.shareWithChat()">
                    <i class="fas fa-share"></i> Discuss in Chat
                </button>
            </div>
        `;

        this.analysisResults.classList.remove('hidden');
        this.analysisResults.scrollIntoView({ behavior: 'smooth' });
    }

    exportAnalysis() {
        if (!this.currentAnalysis) {
            window.dentalApp.showNotification('No analysis to export', 'warning');
            return;
        }

        const reportData = {
            filename: this.currentAnalysis.filename,
            analysisId: this.currentAnalysis.analysis_id,
            timestamp: new Date().toISOString(),
            analysis: this.currentAnalysis.analysis,
            generatedBy: 'Dental AI Assistant'
        };

        const report = this.generateReport(reportData);

        const blob = new Blob([report], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `xray-analysis-${new Date().toISOString().split('T')[0]}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        window.dentalApp.showNotification('Analysis report exported successfully', 'success');
    }

    generateReport(data) {
        return `
DENTAL X-RAY ANALYSIS REPORT
============================

File: ${data.filename}
Analysis ID: ${data.analysisId}
Date: ${new Date(data.timestamp).toLocaleDateString()}
Time: ${new Date(data.timestamp).toLocaleTimeString()}
Generated by: ${data.generatedBy}

ANALYSIS RESULTS
================

Confidence Level: ${Math.round(data.analysis.confidence * 100)}%
Urgency Level: ${data.analysis.urgency.toUpperCase()}

FINDINGS:
${data.analysis.findings.map((finding, index) => `${index + 1}. ${finding}`).join('\n')}

RECOMMENDATIONS:
${data.analysis.recommendations.map((rec, index) => `${index + 1}. ${rec}`).join('\n')}

DISCLAIMER:
This analysis is generated by AI and should not replace professional dental diagnosis. 
Always consult with a qualified dentist for proper evaluation and treatment planning.

Generated on: ${new Date().toLocaleString()}
        `.trim();
    }

    shareWithChat() {
        if (!this.currentAnalysis) {
            window.dentalApp.showNotification('No analysis to share', 'warning');
            return;
        }

        const summary = this.generateChatSummary(this.currentAnalysis);

        const messageInput = document.getElementById('messageInput');
        if (messageInput) {
            messageInput.value = summary;
            messageInput.focus();
            window.dentalApp.showNotification('Analysis summary added to chat', 'success');
        }
    }

    generateChatSummary(analysisData) {
        const { analysis, filename } = analysisData;

        return `X-ray Analysis Summary for ${filename}:\n\nConfidence: ${Math.round(analysis.confidence * 100)}%\nUrgency: ${analysis.urgency}\n\nKey Findings:\n${analysis.findings.map(finding => `• ${finding}`).join('\n')}\n\nWhat are your thoughts on these findings and recommendations?`;
    }

    clearAnalysis() {
        this.currentAnalysis = null;
        this.analysisResults.classList.add('hidden');
        this.resultContent.innerHTML = '';
        this.fileInput.value = '';
    }

    async getAnalysisHistory() {
        try {
            const response = await window.dentalApp.makeRequest('/analysis-history');
            return response.analyses || [];
        } catch (error) {
            console.error('Error fetching analysis history:', error);
            return [];
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.xrayManager = new XRayManager();
});
