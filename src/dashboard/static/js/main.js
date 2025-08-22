document.addEventListener('DOMContentLoaded', function () {
  const yearEl = document.querySelector('.site-footer .container p');
  if (yearEl && yearEl.textContent.includes('{{')) {
    // When not rendered through Flask, fallback to client-side year
    yearEl.textContent = `© ${new Date().getFullYear()} Apex Trading Intelligence`;
  }
});


