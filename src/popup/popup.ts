// Popup script for μDeep Blocker extension
console.log('μDeep Blocker popup script loaded');

// Simple popup script for initial build
document.addEventListener('DOMContentLoaded', () => {
  console.log('μDeep Blocker popup initialized');
  
  const blockedCount = document.getElementById('blockedCount');
  const totalCount = document.getElementById('totalCount');
  const blockRate = document.getElementById('blockRate');
  
  if (blockedCount) blockedCount.textContent = '0';
  if (totalCount) totalCount.textContent = '0';
  if (blockRate) blockRate.textContent = '0%';
});
