/**
 * Liquid & Glass UI/UX Interactions for Streamlit
 */

document.addEventListener('DOMContentLoaded', () => {
    
    // We are running inside a Streamlit iframe component, so we must target the parent document!
    const parentDoc = window.parent.document;
    
    // 0. Inject Background into the very root body so it's not constrained by Streamlit divs
    if (!parentDoc.querySelector('.gradient-bg')) {
        const bgHTML = `
        <div class="gradient-bg">
            <svg xmlns="http://www.w3.org/2000/svg" style="display:none;">
                <defs>
                    <filter id="goo">
                        <feGaussianBlur in="SourceGraphic" stdDeviation="10" result="blur" />
                        <feColorMatrix in="blur" mode="matrix" values="1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 18 -8" result="goo" />
                        <feBlend in="SourceGraphic" in2="goo" />
                    </filter>
                </defs>
            </svg>
            <div class="gradients-container">
                <div class="g1"></div>
                <div class="g2"></div>
                <div class="g3"></div>
                <div class="g4"></div>
                <div class="g5"></div>
                <div class="interactive"></div>
            </div>
        </div>`;
        parentDoc.body.insertAdjacentHTML('afterbegin', bgHTML);
    }
    
    // 1. LIQUID CURSOR EFFECT
    const interBubble = parentDoc.querySelector('.interactive');
    let curX = 0;
    let curY = 0;
    let tgX = 0;
    let tgY = 0;

    function move() {
        if(interBubble) {
            curX += (tgX - curX) / 20; // Smooth easing (liquid feel)
            curY += (tgY - curY) / 20;
            interBubble.style.transform = `translate(${Math.round(curX)}px, ${Math.round(curY)}px)`;
        }
        window.parent.requestAnimationFrame(move);
    }

    // Attach mouse move listener to the parent window
    window.parent.addEventListener('mousemove', (event) => {
        tgX = event.clientX;
        tgY = event.clientY;
    });

    move();

    // 2. APPLE-STYLE SCROLL REVEAL (Targeting Streamlit layout blocks)
    // We poll briefly because Streamlit elements might render after our iframe loads
    setTimeout(() => {
        // 3. GOOGLE MATERIAL RIPPLE EFFECT ON STREAMLIT BUTTONS
        const buttons = parentDoc.querySelectorAll('.stButton > button');
        
        buttons.forEach(btn => {
            btn.addEventListener('mousedown', function (e) {
                const rect = e.target.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                
                const ripple = parentDoc.createElement('span');
                ripple.classList.add('ripple');
                
                const size = Math.max(btn.clientWidth, btn.clientHeight);
                ripple.style.width = ripple.style.height = `${size}px`;
                ripple.style.left = `${x - size/2}px`;
                ripple.style.top = `${y - size/2}px`;
                
                const existingRipple = btn.querySelector('.ripple');
                if(existingRipple) existingRipple.remove();

                this.appendChild(ripple);
                
                setTimeout(() => {
                    ripple.remove();
                }, 600);
            });
        });
    }, 1000); // 1 sec delay to ensure parent DOM is fully populated
});
