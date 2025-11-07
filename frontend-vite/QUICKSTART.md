# 🚀 QUICK START GUIDE - HealthTech AI Frontend

## 📋 **Prerequisites**

- **Node.js** 18+ (https://nodejs.org)
- **pnpm** (recommended) or **npm**
  ```powershell
  npm install -g pnpm
  ```

---

## ⚡ **ONE-COMMAND SETUP**

```powershell
cd frontend-vite
./start.ps1
```

This will:
1. ✅ Check for Node.js & package manager
2. 📦 Install all dependencies
3. 🚀 Start dev server at http://localhost:3000

---

## 🛠️ **Manual Setup**

### **1. Navigate to Frontend**
```powershell
cd d:\Downloads\HealthTech\mvp-healthtech\frontend-vite
```

### **2. Install Dependencies**
```powershell
# Using pnpm (faster)
pnpm install

# OR using npm
npm install
```

### **3. Start Development Server**
```powershell
# Using pnpm
pnpm dev

# OR using npm
npm run dev
```

### **4. Open in Browser**
Visit: **http://localhost:3000**

---

## 📦 **Available Scripts**

| Command | Description |
|---------|-------------|
| `pnpm dev` | Start development server (port 3000) |
| `pnpm build` | Build for production |
| `pnpm preview` | Preview production build |
| `pnpm lint` | Run ESLint |

---

## 🎨 **What You'll See**

### **🏠 Homepage**
- **Hero Section** with animated 3D sphere (Three.js)
- **Kinetic Typography** with shimmer effects
- **Bento Grid** layout for features
- **Smooth Scroll** (Lenis)
- **Magnetic Cursor** (desktop only)
- **Glass Morphism** cards
- **Low-Light** dark mode

### **🎯 Interactive Elements**
- **Magnetic Buttons** - Hover to see attraction effect
- **Auto-Hide Navbar** - Hides on scroll down
- **Theme Toggle** - Switch between dark/light
- **Language Toggle** - English ↔ Arabic (RTL)
- **Scroll Progress** - Top bar shows progress
- **Parallax Effects** - Background animations

---

## 🌟 **2025/2026 Trends Showcase**

✅ **Bento Grids** - Modular, compartmentalized layouts
✅ **3D Immersive** - Three.js powered 3D sphere
✅ **Kinetic Typography** - Animated, motion-driven text
✅ **Blur & Grain** - Tactile texture overlays
✅ **Low Light Mode** - Muted, calming dark theme
✅ **Glass Morphism** - Frosted glass with backdrop blur
✅ **Shimmer Effects** - Animated gradient text
✅ **Magnetic Interactions** - Buttons with magnetic pull
✅ **Scroll Parallax** - Depth through scroll
✅ **Custom Cursor** - Magnetic cursor following

---

## 🎨 **Theme System**

### **Colors**
- **Light Mode**: Clean, minimal (white backgrounds)
- **Dark Mode**: Low-light (muted blacks, soft grays)
- **Accent**: Cyan-blue gradient

### **Toggle Theme**
- Click the **Sun/Moon** icon in navbar
- Persists in localStorage

### **Toggle Language**
- Click the **Globe** icon in navbar
- Switches between English & Arabic (RTL)

---

## 🖱️ **Interactive Features**

### **Custom Cursor (Desktop)**
- Follows your mouse
- Expands on hover over buttons/links
- Magnetic attraction to interactive elements

### **Magnetic Buttons**
- Hover near them
- They'll "pull" towards your cursor

### **Smooth Scroll**
- Momentum-based scrolling
- Natural physics easing

### **Auto-Hide Navbar**
- Scroll down → Navbar hides
- Scroll up → Navbar appears

---

## 📱 **Responsive Design**

- ✅ Mobile-first approach
- ✅ Tablet optimized
- ✅ Desktop enhanced (custom cursor, parallax)
- ✅ Reduced motion support

---

## ♿ **Accessibility**

- ✅ `prefers-reduced-motion` support
- ✅ Focus-visible states
- ✅ Semantic HTML
- ✅ ARIA labels
- ✅ Keyboard navigation

---

## 🎯 **Pages Structure**

| Page | Route | Status |
|------|-------|--------|
| Home | `/` | ✅ Complete with all trends |
| Features | `/features` | 📝 Placeholder |
| Clinical Notes | `/features/clinical-notes` | 📝 Placeholder |
| Voice Transcription | `/features/voice-transcription` | 📝 Placeholder |
| FHIR Integration | `/features/fhir-integration` | 📝 Placeholder |
| SOAP Generation | `/features/soap-generation` | 📝 Placeholder |
| Dashboard | `/dashboard` | 📝 Placeholder |
| About | `/about` | 📝 Placeholder |
| Pricing | `/pricing` | 📝 Placeholder |
| Demo | `/demo` | 📝 Placeholder |

---

## 🐛 **Troubleshooting**

### **Port 3000 already in use?**
```powershell
# Kill process on port 3000
Stop-Process -Id (Get-NetTCPConnection -LocalPort 3000).OwningProcess -Force

# Or use different port
pnpm dev --port 3001
```

### **Module not found errors?**
```powershell
# Clean install
Remove-Item node_modules -Recurse -Force
Remove-Item package-lock.json -Force  # or pnpm-lock.yaml
pnpm install
```

### **TypeScript errors?**
These are expected before `npm install` runs. They'll disappear after installing dependencies.

---

## 🔥 **Performance Tips**

1. **Code Splitting** - Routes automatically split
2. **Lazy Loading** - Components load on demand
3. **Image Optimization** - Use SVG or WebP
4. **Reduce Motion** - Respects user preferences

---

## 🎨 **Customization**

### **Colors** (`tailwind.config.js`)
```js
theme: {
  extend: {
    colors: {
      accent: { /* your colors */ },
      medical: { /* your colors */ }
    }
  }
}
```

### **Animations** (`globals.css`)
```css
@keyframes your-animation {
  /* keyframes */
}
```

---

## 🚀 **Build for Production**

```powershell
pnpm build
```

Output: `dist/` folder

### **Preview Production Build**
```powershell
pnpm preview
```

---

## 📚 **Tech Stack**

| Category | Technology |
|----------|-----------|
| **Framework** | React 18 |
| **Build Tool** | Vite 5 |
| **Styling** | Tailwind CSS v4 |
| **Animations** | Framer Motion |
| **3D** | Three.js + React Three Fiber |
| **Scroll** | Lenis |
| **State** | Zustand |
| **Router** | React Router v6 |
| **Icons** | Lucide React |

---

## 💡 **Next Steps**

1. ✅ Run the app
2. 🎨 Explore the design
3. 📝 Add content to placeholder pages
4. 🔌 Connect to backend APIs
5. 📊 Add analytics
6. 🧪 Write tests
7. 🚀 Deploy!

---

## 🤝 **Need Help?**

- **Vite Docs**: https://vitejs.dev
- **Tailwind Docs**: https://tailwindcss.com
- **Framer Motion**: https://www.framer.com/motion
- **Three.js**: https://threejs.org

---

## 🎉 **You're All Set!**

Your ultra-modern, trend-setting frontend is ready to impress! 🚀

**Features implemented:**
- ✅ All 2025/2026 UI/UX trends
- ✅ Bento grids, 3D, kinetic typography
- ✅ Low-light mode, glass morphism
- ✅ Magnetic interactions, smooth scroll
- ✅ Dark/light themes, Arabic/RTL
- ✅ Accessibility & performance optimized

**Now go blow your client's mind!** 💥
