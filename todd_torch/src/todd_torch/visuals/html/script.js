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

const toddOverflow = () => {
  const canvas = document.querySelector('.todd-canvas');
  const { left, top, right, bottom } = canvas.getBoundingClientRect();
  return Array.from(document.querySelectorAll('.todd-semantic'))
    .find((element) => {
      const bounds = element.getBoundingClientRect();
      return element.scrollWidth > element.clientWidth + 1 ||
        element.scrollHeight > element.clientHeight + 1 ||
        bounds.left < left - 1 || bounds.top < top - 1 ||
        bounds.right > right + 1 || bounds.bottom > bottom + 1;
    })?.className;
};

const render = async () => {
  try {
    const errors = [];
    document.querySelectorAll('.todd-latex').forEach((element) => {
      try {
        katex.render(element.textContent, element, {
          displayMode: element.dataset.toddDisplayMode === 'true',
        });
      } catch (error) {
        element.textContent = String(error);
        element.style.color = '#b42318';
        errors.push(String(error));
      }
    });
    await document.fonts.ready;
    await Promise.all(Array.from(document.images, (image) => image.decode()));
    fitTextElements();
    const overflow = toddOverflow();
    if (overflow) {
      errors.push(`Semantic content exceeds the HTML canvas: ${overflow}`);
    }
    if (errors.length) {
      return errors.join('\n');
    }
    return '';
  } catch (error) {
    return String(error);
  }
};

let renderPromise;
window.todd_render = () => {
  renderPromise ??= render();
  return renderPromise;
};

window.addEventListener('load', () => { void window.todd_render(); });
