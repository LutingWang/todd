const fitText = (element) => {
  const fontSize = Number(element.dataset.toddFontSize);
  let currentSize = fontSize;
  element.style.fontSize = `${currentSize}px`;
  while (
    currentSize > 1 &&
    (element.scrollWidth > element.clientWidth ||
      element.scrollHeight > element.clientHeight)
  ) {
    currentSize -= 1;
    element.style.fontSize = `${currentSize}px`;
  }
};

const fitTextElements = () => {
  const observer = new ResizeObserver((entries) => {
    entries.forEach((entry) => fitText(entry.target));
  });
  document.querySelectorAll('[data-todd-fit-text]').forEach((element) => {
    fitText(element);
    observer.observe(element);
  });
};

window.addEventListener('load', fitTextElements);
