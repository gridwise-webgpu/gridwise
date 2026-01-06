// Modern Gridwise Documentation - Interactive Features

(function() {
  'use strict';

  // Sidebar Toggle Functionality
  function initSidebarToggle() {
    const sidebar = document.getElementById('docsSidebar');
    const toggleBtn = document.getElementById('sidebarToggle');
    const menuIcon = document.getElementById('menuIcon');
    const closeIcon = document.getElementById('closeIcon');

    if (!sidebar || !toggleBtn) return;

    // Get saved state or default to open on desktop
    const savedState = localStorage.getItem('sidebarOpen');
    const isOpen = savedState !== null ? savedState === 'true' : window.innerWidth > 768;

    // Set initial state
    if (!isOpen) {
      sidebar.classList.add('closed');
      menuIcon?.classList.remove('hidden');
      closeIcon?.classList.add('hidden');
    } else {
      sidebar.classList.remove('closed');
      menuIcon?.classList.add('hidden');
      closeIcon?.classList.remove('hidden');
    }

    // Toggle sidebar on button click
    toggleBtn.addEventListener('click', function() {
      const isClosed = sidebar.classList.contains('closed');
      
      if (isClosed) {
        sidebar.classList.remove('closed');
        sidebar.classList.add('open');
        menuIcon?.classList.add('hidden');
        closeIcon?.classList.remove('hidden');
        localStorage.setItem('sidebarOpen', 'true');
      } else {
        sidebar.classList.add('closed');
        sidebar.classList.remove('open');
        menuIcon?.classList.remove('hidden');
        closeIcon?.classList.add('hidden');
        localStorage.setItem('sidebarOpen', 'false');
      }
    });

    // Handle responsive behavior
    window.addEventListener('resize', function() {
      if (window.innerWidth > 768 && sidebar.classList.contains('closed')) {
        const savedState = localStorage.getItem('sidebarOpen');
        if (savedState === 'true') {
          sidebar.classList.remove('closed');
        }
      }
    });
  }

  // Code Block Copy Functionality
  function initCodeCopyButtons() {
    // Find all code blocks
    const codeBlocks = document.querySelectorAll('pre');

    codeBlocks.forEach(function(block) {
      // Skip if already processed
      if (block.parentElement.classList.contains('code-block')) return;

      // Wrap the pre element in a code-block div
      const wrapper = document.createElement('div');
      wrapper.className = 'code-block';
      block.parentNode.insertBefore(wrapper, block);
      wrapper.appendChild(block);

      // Create copy button
      const copyButton = document.createElement('button');
      copyButton.className = 'copy-button';
      copyButton.textContent = 'Copy';
      copyButton.setAttribute('aria-label', 'Copy code to clipboard');

      // Add copy functionality
      copyButton.addEventListener('click', function() {
        const code = block.querySelector('code')?.textContent || block.textContent;
        
        navigator.clipboard.writeText(code).then(function() {
          copyButton.textContent = 'Copied!';
          copyButton.style.backgroundColor = '#10b981';
          
          setTimeout(function() {
            copyButton.textContent = 'Copy';
            copyButton.style.backgroundColor = '';
          }, 2000);
        }).catch(function(err) {
          console.error('Failed to copy:', err);
          copyButton.textContent = 'Error';
          setTimeout(function() {
            copyButton.textContent = 'Copy';
          }, 2000);
        });
      });

      wrapper.appendChild(copyButton);
    });
  }

  // Search Functionality
  function initSearch() {
    const searchInput = document.getElementById('docsSearch');
    const navLinks = document.querySelectorAll('.nav-section a');

    if (!searchInput) return;

    searchInput.addEventListener('input', function(e) {
      const query = e.target.value.toLowerCase();

      navLinks.forEach(function(link) {
        const text = link.textContent.toLowerCase();
        const listItem = link.parentElement;

        if (text.includes(query)) {
          listItem.style.display = '';
        } else {
          listItem.style.display = 'none';
        }
      });

      // Hide sections if all items are hidden
      document.querySelectorAll('.nav-section').forEach(function(section) {
        const visibleItems = Array.from(section.querySelectorAll('li')).filter(
          li => li.style.display !== 'none'
        );
        
        if (visibleItems.length === 0) {
          section.style.display = 'none';
        } else {
          section.style.display = '';
        }
      });
    });
  }

  // Smooth Scroll for Anchor Links
  function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(function(anchor) {
      anchor.addEventListener('click', function(e) {
        const href = this.getAttribute('href');
        if (href === '#') return;

        const target = document.querySelector(href);
        if (target) {
          e.preventDefault();
          target.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
          });
        }
      });
    });
  }

  // Preserve Sidebar Scroll Position
  function initSidebarScrollPersistence() {
    const sidebarContent = document.querySelector('.sidebar-content');
    if (!sidebarContent) return;

    // Restore scroll position on page load
    const savedScrollPos = sessionStorage.getItem('sidebarScrollPos');
    if (savedScrollPos) {
      // Use setTimeout to ensure DOM is fully rendered
      setTimeout(function() {
        sidebarContent.scrollTop = parseInt(savedScrollPos, 10);
      }, 0);
    }

    // Save scroll position on scroll
    let scrollTimer;
    sidebarContent.addEventListener('scroll', function() {
      clearTimeout(scrollTimer);
      scrollTimer = setTimeout(function() {
        sessionStorage.setItem('sidebarScrollPos', sidebarContent.scrollTop);
      }, 50);
    });

    // Save scroll position before navigating away
    const sidebarLinks = sidebarContent.querySelectorAll('a');
    sidebarLinks.forEach(function(link) {
      link.addEventListener('click', function(e) {
        // Save current scroll position
        sessionStorage.setItem('sidebarScrollPos', sidebarContent.scrollTop);
      });
    });

    // Also save on page unload
    window.addEventListener('beforeunload', function() {
      sessionStorage.setItem('sidebarScrollPos', sidebarContent.scrollTop);
    });
  }

  // Highlight Active Section on Scroll
  function initScrollSpy() {
    const sections = document.querySelectorAll('.content-section[id]');
    const navLinks = document.querySelectorAll('.nav-section a');

    if (sections.length === 0) return;

    function highlightNavigation() {
      let current = '';
      
      sections.forEach(function(section) {
        const sectionTop = section.offsetTop;
        const sectionHeight = section.clientHeight;
        
        if (window.pageYOffset >= sectionTop - 200) {
          current = section.getAttribute('id');
        }
      });

      navLinks.forEach(function(link) {
        link.classList.remove('active');
        const href = link.getAttribute('href');
        if (href && href.includes('#' + current)) {
          link.classList.add('active');
        }
      });
    }

    window.addEventListener('scroll', highlightNavigation);
    highlightNavigation();
  }

  // Initialize all features when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function() {
      initSidebarToggle();
      initCodeCopyButtons();
      initSearch();
      initSmoothScroll();
      initSidebarScrollPersistence();
      initScrollSpy();
      initDropdownMenus();
    });
  } else {
    initSidebarToggle();
    initCodeCopyButtons();
    initSearch();
    initSmoothScroll();
    initSidebarScrollPersistence();
    initScrollSpy();
    initDropdownMenus();
  }

  // Dropdown Menus for Sidebar Sections
  function initDropdownMenus() {
    const sectionTitles = document.querySelectorAll('.nav-section-title');
    
    sectionTitles.forEach(title => {
      // Add chevron icon if not already present
      if (!title.querySelector('svg')) {
        const chevron = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        chevron.setAttribute('fill', 'none');
        chevron.setAttribute('viewBox', '0 0 24 24');
        chevron.setAttribute('stroke', 'currentColor');
        chevron.innerHTML = '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />';
        title.appendChild(chevron);
      }

      // Get the section and its list
      const section = title.parentElement;
      const list = section.querySelector('ul');
      if (!list) return;

      // Get saved state from localStorage
      const sectionId = title.textContent.trim().replace(/\s+/g, '-').toLowerCase();
      const savedState = localStorage.getItem(`nav-section-${sectionId}`);
      const isCollapsed = savedState === 'collapsed';

      // Set initial state
      if (isCollapsed) {
        title.classList.add('collapsed');
        list.classList.add('collapsed');
      }

      // Toggle on click
      title.addEventListener('click', function(e) {
        e.preventDefault();
        const isCurrentlyCollapsed = list.classList.contains('collapsed');
        
        if (isCurrentlyCollapsed) {
          title.classList.remove('collapsed');
          list.classList.remove('collapsed');
          localStorage.setItem(`nav-section-${sectionId}`, 'expanded');
        } else {
          title.classList.add('collapsed');
          list.classList.add('collapsed');
          localStorage.setItem(`nav-section-${sectionId}`, 'collapsed');
        }
      });
    });
  }
})();
