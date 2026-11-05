import { jsPDF } from 'jspdf';
import devanagariFontUrl from '@expo-google-fonts/noto-sans-devanagari/400Regular/NotoSansDevanagari_400Regular.ttf?url';

function formatTimestamp(value) {
  if (!value) return 'Unknown';
  try {
    return new Date(value).toLocaleString();
  } catch {
    return String(value);
  }
}

function severityColor(severity) {
  return severity === 'high' ? [198, 40, 40] : [239, 108, 0];
}

async function addDevanagariFont(doc) {
  const response = await fetch(devanagariFontUrl);
  if (!response.ok) throw new Error(`Unable to load PDF font: ${response.status}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  let binary = '';
  const chunkSize = 0x8000;
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + chunkSize));
  }
  doc.addFileToVFS('NotoSansDevanagari.ttf', btoa(binary));
  doc.addFont('NotoSansDevanagari.ttf', 'NotoSansDevanagari', 'normal');
}

export async function exportIncidentsPDF(incidents) {
  const doc = new jsPDF({ unit: 'mm', format: 'a4' });
  await addDevanagariFont(doc);
  const pageWidth = doc.internal.pageSize.getWidth();
  const pageHeight = doc.internal.pageSize.getHeight();
  const margin = 16;
  const contentWidth = pageWidth - margin * 2;
  let y = margin;

  const pageBreak = (height = 12) => {
    if (y + height > pageHeight - margin) {
      doc.addPage();
      y = margin;
    }
  };

  doc.setProperties({
    title: 'SurakshaNet Incident Report',
    subject: 'Local WhatsApp Web abuse detection incidents',
    creator: 'SurakshaNet',
  });

  doc.setFont('helvetica', 'bold');
  doc.setFontSize(18);
  doc.text('SurakshaNet Incident Report', pageWidth / 2, y, { align: 'center' });
  y += 8;
  doc.setFont('helvetica', 'normal');
  doc.setFontSize(9);
  doc.text('Local WhatsApp Web abuse detection log', pageWidth / 2, y, { align: 'center' });
  y += 12;

  doc.setFontSize(10);
  doc.text(`Generated: ${formatTimestamp(new Date().toISOString())}`, margin, y);
  y += 5;
  doc.text(`Incidents: ${incidents.length}`, margin, y);
  y += 8;
  doc.setDrawColor(210, 210, 210);
  doc.line(margin, y, pageWidth - margin, y);
  y += 8;

  incidents.forEach((incident, index) => {
    pageBreak(42);

    const severity = String(incident.severity || 'medium').toUpperCase();
    const [r, g, b] = severityColor(String(incident.severity).toLowerCase());
    doc.setFillColor(245, 245, 245);
    doc.rect(margin, y, contentWidth, 7, 'F');
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(10);
    doc.setTextColor(0, 0, 0);
    doc.text(`Incident ${index + 1}`, margin + 2, y + 4.8);
    doc.setFillColor(r, g, b);
    doc.rect(pageWidth - margin - 25, y + 1, 23, 5, 'F');
    doc.setFontSize(7);
    doc.setTextColor(255, 255, 255);
    doc.text(severity, pageWidth - margin - 13.5, y + 4.5, { align: 'center' });
    doc.setTextColor(0, 0, 0);
    y += 11;

    doc.setFont('NotoSansDevanagari', 'normal');
    doc.setFontSize(9);
    doc.text(`Time: ${formatTimestamp(incident.ts)}`, margin + 2, y);
    y += 5;
    doc.text(`Contact: ${incident.contact || 'Unknown chat'}`, margin + 2, y);
    y += 5;
    doc.text(`Confidence: ${((Number(incident.score) || 0) * 100).toFixed(0)}%`, margin + 2, y);
    y += 7;

    doc.setFont('helvetica', 'bold');
    doc.text('Message:', margin + 2, y);
    y += 4;
    doc.setFont('NotoSansDevanagari', 'normal');
    const lines = doc.splitTextToSize(String(incident.text || ''), contentWidth - 8);
    let offset = 0;
    while (offset < lines.length) {
      pageBreak(10);
      const availableLines = Math.max(1, Math.floor((pageHeight - margin - y - 6) / 4));
      const chunk = lines.slice(offset, offset + availableLines);
      doc.text(chunk, margin + 4, y);
      y += chunk.length * 4 + 7;
      offset += chunk.length;
    }

    doc.setFontSize(7);
    doc.setTextColor(110, 110, 110);
    doc.text(`ID: ${String(incident.id || '').slice(0, 24)}`, margin + 2, y);
    doc.setTextColor(0, 0, 0);
    y += 10;
  });

  const pages = doc.internal.getNumberOfPages();
  for (let page = 1; page <= pages; page += 1) {
    doc.setPage(page);
    doc.setFont('helvetica', 'italic');
    doc.setFontSize(7);
    doc.setTextColor(110, 110, 110);
    doc.text(
      `SurakshaNet · Page ${page}/${pages} · Local report`,
      pageWidth / 2,
      pageHeight - 7,
      { align: 'center' },
    );
  }

  doc.save(`surakshanet-incidents-${new Date().toISOString().slice(0, 10)}.pdf`);
}
