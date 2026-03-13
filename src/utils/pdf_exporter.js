/**
 * PDF Exporter - Court-Admissible Evidence Reports
 * Generates professional legal documents with evidence metadata and screenshots
 */

import { jsPDF } from 'jspdf';

/**
 * Generate PDF evidence report
 * @param {Array} evidenceList - List of evidence records
 * @param {Object} options - PDF generation options
 * @returns {jsPDF} PDF document object
 */
export function generateEvidenceReport(evidenceList, options = {}) {
    const doc = new jsPDF({
        orientation: 'portrait',
        unit: 'mm',
        format: 'a4'
    });

    const pageWidth = doc.internal.pageSize.getWidth();
    const pageHeight = doc.internal.pageSize.getHeight();
    const margin = 20;
    const contentWidth = pageWidth - (2 * margin);

    let yPosition = margin;

    // Helper function to add new page if needed
    const checkPageBreak = (requiredSpace = 20) => {
        if (yPosition + requiredSpace > pageHeight - margin) {
            doc.addPage();
            yPosition = margin;
            return true;
        }
        return false;
    };

    // Header
    doc.setFontSize(22);
    doc.setFont('helvetica', 'bold');
    doc.text('SURAKSHANET EVIDENCE REPORT', pageWidth / 2, yPosition, { align: 'center' });
    yPosition += 10;

    doc.setFontSize(10);
    doc.setFont('helvetica', 'normal');
    doc.text('Court-Admissible Digital Evidence Documentation', pageWidth / 2, yPosition, { align: 'center' });
    yPosition += 15;

    // Report metadata
    doc.setFontSize(10);
    doc.setFont('helvetica', 'bold');
    doc.text('Report Generated:', margin, yPosition);
    doc.setFont('helvetica', 'normal');
    doc.text(new Date().toLocaleString('en-IN', { timeZone: 'Asia/Kolkata' }), margin + 50, yPosition);
    yPosition += 7;

    doc.setFont('helvetica', 'bold');
    doc.text('Total Incidents:', margin, yPosition);
    doc.setFont('helvetica', 'normal');
    doc.text(evidenceList.length.toString(), margin + 50, yPosition);
    yPosition += 7;

    doc.setFont('helvetica', 'bold');
    doc.text('Report ID:', margin, yPosition);
    doc.setFont('helvetica', 'normal');
    doc.text(generateReportId(), margin + 50, yPosition);
    yPosition += 15;

    // Horizontal line
    doc.setLineWidth(0.5);
    doc.line(margin, yPosition, pageWidth - margin, yPosition);
    yPosition += 10;

    // Evidence summary
    doc.setFontSize(14);
    doc.setFont('helvetica', 'bold');
    doc.text('EVIDENCE SUMMARY', margin, yPosition);
    yPosition += 10;

    const severityCounts = countBySeverity(evidenceList);
    doc.setFontSize(10);
    doc.setFont('helvetica', 'normal');

    Object.entries(severityCounts).forEach(([severity, count]) => {
        if (count > 0) {
            doc.text(`${severity} Severity: ${count} incident(s)`, margin + 5, yPosition);
            yPosition += 6;
        }
    });
    yPosition += 10;

    // Individual evidence entries
    doc.setFontSize(14);
    doc.setFont('helvetica', 'bold');
    doc.text('DETAILED EVIDENCE', margin, yPosition);
    yPosition += 10;

    evidenceList.forEach((evidence, index) => {
        checkPageBreak(60);

        // Evidence header
        doc.setFillColor(240, 240, 240);
        doc.rect(margin, yPosition, contentWidth, 8, 'F');

        doc.setFontSize(12);
        doc.setFont('helvetica', 'bold');
        doc.text(`Incident #${index + 1}`, margin + 2, yPosition + 5.5);

        // Severity badge
        const severityColor = getSeverityColor(evidence.severity);
        doc.setFillColor(severityColor.r, severityColor.g, severityColor.b);
        doc.rect(pageWidth - margin - 30, yPosition + 1, 28, 6, 'F');
        doc.setTextColor(255, 255, 255);
        doc.setFontSize(9);
        doc.text(evidence.severity || 'UNKNOWN', pageWidth - margin - 16, yPosition + 5, { align: 'center' });
        doc.setTextColor(0, 0, 0);

        yPosition += 12;

        // Evidence details
        doc.setFontSize(9);
        doc.setFont('helvetica', 'bold');

        doc.text('Evidence ID:', margin + 2, yPosition);
        doc.setFont('helvetica', 'normal');
        doc.text(evidence.id || 'N/A', margin + 30, yPosition);
        yPosition += 5;

        doc.setFont('helvetica', 'bold');
        doc.text('Timestamp:', margin + 2, yPosition);
        doc.setFont('helvetica', 'normal');
        doc.text(formatTimestamp(evidence.timestamp), margin + 30, yPosition);
        yPosition += 5;

        doc.setFont('helvetica', 'bold');
        doc.text('Source:', margin + 2, yPosition);
        doc.setFont('helvetica', 'normal');
        doc.text(evidence.source || 'Unknown', margin + 30, yPosition);
        yPosition += 5;

        doc.setFont('helvetica', 'bold');
        doc.text('Detection Score:', margin + 2, yPosition);
        doc.setFont('helvetica', 'normal');
        doc.text(`${(evidence.maxScore * 100).toFixed(1)}%`, margin + 30, yPosition);
        yPosition += 5;

        // Categories
        if (evidence.categories && evidence.categories.length > 0) {
            doc.setFont('helvetica', 'bold');
            doc.text('Categories:', margin + 2, yPosition);
            doc.setFont('helvetica', 'normal');

            const categories = evidence.categories
                .map(c => `${c.category.toUpperCase()} (${(c.score * 100).toFixed(0)}%)`)
                .join(', ');

            const categoryLines = doc.splitTextToSize(categories, contentWidth - 32);
            doc.text(categoryLines, margin + 30, yPosition);
            yPosition += categoryLines.length * 5;
        }

        yPosition += 3;

        // Content
        doc.setFont('helvetica', 'bold');
        doc.text('Content:', margin + 2, yPosition);
        yPosition += 5;

        doc.setFont('helvetica', 'normal');
        doc.setFontSize(8);

        // Text box for content
        doc.setDrawColor(200, 200, 200);
        doc.setFillColor(250, 250, 250);

        const textLines = doc.splitTextToSize(evidence.text || 'No content', contentWidth - 6);
        const textBoxHeight = Math.min(textLines.length * 4 + 4, 40);

        checkPageBreak(textBoxHeight + 5);

        doc.rect(margin + 2, yPosition, contentWidth - 4, textBoxHeight, 'FD');
        doc.text(textLines, margin + 4, yPosition + 3);
        yPosition += textBoxHeight + 5;

        // Screenshot placeholder (if available)
        if (evidence.screenshot) {
            checkPageBreak(60);

            doc.setFontSize(9);
            doc.setFont('helvetica', 'bold');
            doc.text('Visual Evidence:', margin + 2, yPosition);
            yPosition += 5;

            try {
                // Add screenshot to PDF
                const imgWidth = contentWidth - 4;
                const imgHeight = 50; // Fixed height for consistency

                doc.addImage(evidence.screenshot, 'PNG', margin + 2, yPosition, imgWidth, imgHeight);
                yPosition += imgHeight + 5;
            } catch (error) {
                console.error('Failed to add screenshot to PDF:', error);
                doc.setFont('helvetica', 'italic');
                doc.text('[Screenshot unavailable]', margin + 4, yPosition);
                yPosition += 5;
            }
        }

        yPosition += 10;

        // Separator line
        if (index < evidenceList.length - 1) {
            checkPageBreak(5);
            doc.setDrawColor(200, 200, 200);
            doc.setLineWidth(0.3);
            doc.line(margin, yPosition, pageWidth - margin, yPosition);
            yPosition += 10;
        }
    });

    // Footer on each page
    const totalPages = doc.internal.getNumberOfPages();
    for (let i = 1; i <= totalPages; i++) {
        doc.setPage(i);
        doc.setFontSize(8);
        doc.setFont('helvetica', 'italic');
        doc.setTextColor(128, 128, 128);
        doc.text(
            `SurakshaNet Evidence Report - Page ${i} of ${totalPages} - Generated: ${new Date().toLocaleDateString('en-IN')}`,
            pageWidth / 2,
            pageHeight - 10,
            { align: 'center' }
        );
        doc.setTextColor(0, 0, 0);
    }

    return doc;
}

/**
 * Export evidence to PDF and download
 * @param {Array} evidenceList - Evidence to export
 * @param {string} filename - PDF filename (without extension)
 */
export function exportToPDF(evidenceList, filename = 'surakshanet_evidence_report') {
    const doc = generateEvidenceReport(evidenceList);
    const timestamp = new Date().toISOString().split('T')[0];
    doc.save(`${filename}_${timestamp}.pdf`);

    console.log('📄 PDF exported:', `${filename}_${timestamp}.pdf`);
}

/**
 * Generate unique report ID
 * @returns {string} Report ID
 */
function generateReportId() {
    const timestamp = Date.now().toString(36);
    const random = Math.random().toString(36).substring(2, 7);
    return `SR-${timestamp}-${random}`.toUpperCase();
}

/**
 * Count evidence by severity
 * @param {Array} evidenceList - Evidence list
 * @returns {Object} Severity counts
 */
function countBySeverity(evidenceList) {
    const counts = { HIGH: 0, MEDIUM: 0, LOW: 0, UNKNOWN: 0 };
    evidenceList.forEach(e => {
        counts[e.severity || 'UNKNOWN']++;
    });
    return counts;
}

/**
 * Get color for severity level
 * @param {string} severity - Severity level
 * @returns {Object} RGB color
 */
function getSeverityColor(severity) {
    const colors = {
        HIGH: { r: 211, g: 47, b: 47 },      // Red
        MEDIUM: { r: 245, g: 124, b: 0 },    // Orange
        LOW: { r: 251, g: 192, b: 45 },      // Yellow
        UNKNOWN: { r: 158, g: 158, b: 158 }  // Gray
    };
    return colors[severity] || colors.UNKNOWN;
}

/**
 * Format timestamp for display
 * @param {string} timestamp - ISO timestamp
 * @returns {string} Formatted timestamp
 */
function formatTimestamp(timestamp) {
    if (!timestamp) return 'N/A';
    const date = new Date(timestamp);
    return date.toLocaleString('en-IN', {
        timeZone: 'Asia/Kolkata',
        dateStyle: 'medium',
        timeStyle: 'medium'
    });
}
