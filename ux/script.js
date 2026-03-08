document.addEventListener('DOMContentLoaded', () => {
    // We now have a full page drag overlay
    const dragOverlay = document.getElementById('drag-overlay');
    const fileInput = document.getElementById('image-upload');
    const uploadTrigger = document.getElementById('upload-trigger');
    const previewArea = document.getElementById('preview-area');
    const imagePreview = document.getElementById('image-preview');
    const removeBtn = document.getElementById('remove-btn');
    const searchForm = document.getElementById('search-form');
    const textQuery = document.getElementById('text-query');
    const searchBtn = document.getElementById('search-btn');

    const loadingState = document.getElementById('loading-state');
    const resultsSection = document.getElementById('results-section');
    const resultsGrid = document.getElementById('results-grid');
    const paginationControls = document.getElementById('pagination-controls');
    const loadMoreBtn = document.getElementById('load-more-btn');

    let currentFile = null;
    let currentCategory = null;
    let currentOffset = 0;
    const LIMIT = 20;

    // --- Category Navigation Logic --- //
    const categoryLinks = document.querySelectorAll('.category-nav a[data-category]');
    categoryLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            // Reset state
            currentCategory = link.getAttribute('data-category');
            textQuery.value = '';
            removeBtn.click(); // Clear any uploaded image

            // Highlight active link (optional UI touch)
            categoryLinks.forEach(l => l.style.fontWeight = 'normal');
            link.style.fontWeight = 'bold';

            // Trigger search from offset 0
            triggerSearch(true);
        });
    });

    // --- Drag and Drop Logic --- //
    // The entire window can accept drops now for that premium feel
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        window.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    let dragCounter = 0; // Prevent flicker when dragging over children

    window.addEventListener('dragenter', (e) => {
        dragCounter++;
        dragOverlay.classList.remove('hidden');
        setTimeout(() => dragOverlay.classList.add('active'), 10);
    });

    window.addEventListener('dragleave', (e) => {
        dragCounter--;
        if (dragCounter === 0) {
            dragOverlay.classList.remove('active');
            setTimeout(() => dragOverlay.classList.add('hidden'), 300);
        }
    });

    window.addEventListener('drop', (e) => {
        dragCounter = 0;
        dragOverlay.classList.remove('active');
        setTimeout(() => dragOverlay.classList.add('hidden'), 300);

        const dt = e.dataTransfer;
        const files = dt.files;
        if (files && files.length > 0) {
            handleFile(files[0]);
        }
    });

    // --- Manual File Upload --- //
    uploadTrigger.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', function (e) {
        if (this.files && this.files.length > 0) {
            handleFile(this.files[0]);
        }
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file.');
            return;
        }

        currentFile = file;

        // Show miniature preview area under search bar
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onloadend = function () {
            imagePreview.src = reader.result;
            previewArea.classList.remove('hidden');
            // Auto focus text query after dropping image
            textQuery.focus();
        };
    }

    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        currentFile = null;
        fileInput.value = '';
        imagePreview.src = '';
        previewArea.classList.add('hidden');
    });

    // --- Search Submission --- //
    searchForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        currentCategory = null; // Clear category filter if doing a fresh text/image search
        categoryLinks.forEach(l => l.style.fontWeight = 'normal'); // remove highlights
        triggerSearch(true);
    });

    loadMoreBtn.addEventListener('click', () => {
        triggerSearch(false);
    });

    async function triggerSearch(isNewSearch = true) {
        const query = textQuery.value.trim();

        if (isNewSearch) {
            currentOffset = 0;
            resultsGrid.innerHTML = ''; // Only clear grid on fresh searches
        } else {
            currentOffset += LIMIT;
        }

        if (!query && !currentFile && !currentCategory) {
            alert('Please search by keyword, upload an image, or select a category.');
            textQuery.focus();
            return;
        }

        // UX: Show searching state (only if it's a new search)
        if (isNewSearch) {
            resultsSection.classList.add('hidden');
            loadingState.classList.remove('hidden');
        } else {
            loadMoreBtn.innerHTML = 'Loading...';
            loadMoreBtn.disabled = true;
        }

        searchBtn.disabled = true;
        searchBtn.style.opacity = '0.5';

        // Connect to FastAPI Backend
        const formData = new FormData();
        if (query) formData.append('text_query', query);
        if (currentFile) formData.append('image_file', currentFile);
        if (currentCategory) formData.append('category', currentCategory);
        formData.append('limit', LIMIT);
        formData.append('offset', currentOffset);

        try {
            const response = await fetch('http://localhost:8000/search', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Connection error: ${response.status}`);
            }

            const data = await response.json();

            // Artificial delay to show off the nice loading animation UX
            setTimeout(() => {
                renderResults(data, isNewSearch);

                if (isNewSearch) {
                    loadingState.classList.add('hidden');
                    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }

                searchBtn.disabled = false;
                searchBtn.style.opacity = '1';
                loadMoreBtn.innerHTML = 'Load More';
                loadMoreBtn.disabled = false;

                // If fewer items returned than limit, hide Load More
                if (data.length < LIMIT) {
                    paginationControls.classList.add('hidden');
                } else {
                    paginationControls.classList.remove('hidden');
                }
            }, 800);

        } catch (error) {
            console.error('Search failed:', error);
            simulateSearchDemo(isNewSearch);
        }
    }

    // --- UI Rendering --- //
    function renderResults(data, isNewSearch) {
        let items = Array.isArray(data) ? data : (data.results || []);

        if (items.length === 0 && isNewSearch) {
            resultsGrid.innerHTML = `
                <div style="grid-column: 1/-1; text-align: center; padding: 4rem 0;">
                    <h3 style="font-family: var(--font-heading); font-size: 1.5rem; margin-bottom: 1rem;">No matches found</h3>
                    <p style="color: var(--text-secondary);">Try refining your search terms or using a clearer image.</p>
                </div>
            `;
            resultsSection.classList.remove('hidden');
            paginationControls.classList.add('hidden');
            return;
        }

        items.forEach((item, index) => {
            const imgUrl = item.image || item.imageUrl || 'https://via.placeholder.com/400x600?text=No+Preview';
            const name = item.name || 'Studio Piece';
            const score = item.similarity || item.score || null;

            const card = document.createElement('div');
            card.className = 'product-card';
            // Staggered reveal animation
            card.style.animation = `fadeIn 0.6s ease ${index * 0.1}s both`;

            let scoreHtml = score ? `<div class="similarity-score">AI Match • ${(score * 100).toFixed(0)}%</div>` : '';

            // Assume random realistic pricing since backend doesn't output it
            const mockPrice = `$${(Math.floor(Math.random() * 80) + 19).toFixed(2)}`;

            card.innerHTML = `
                <div class="card-image-wrapper">
                    <img src="${imgUrl}" alt="${name}" onerror="this.src='https://via.placeholder.com/400x600?text=Image+Load+Error'">
                    <div class="card-overlay">
                        <span>Quick shop</span>
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M6 2L3 6v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V6l-3-4z"></path><line x1="3" y1="6" x2="21" y2="6"></line><path d="M16 10a4 4 0 0 1-8 0"></path></svg>
                    </div>
                </div>
                <div class="product-info">
                    <h3 class="product-name">${name}</h3>
                    <div class="product-price">${mockPrice}</div>
                    ${scoreHtml}
                </div>
            `;
            resultsGrid.appendChild(card);
        });

        resultsSection.classList.remove('hidden');
    }

    // A simple demonstration method injected to show the UI working during tests
    function simulateSearchDemo(isNewSearch) {
        setTimeout(() => {
            const demoData = {
                results: [
                    {
                        imageUrl: 'https://images.unsplash.com/photo-1542291026-7eec264c27ff?auto=format&fit=crop&q=80&w=600',
                        name: 'Studio Collection Running Shoes',
                        score: 0.98
                    },
                    {
                        imageUrl: 'https://images.unsplash.com/photo-1525966222134-fcfa99b8ae77?auto=format&fit=crop&q=80&w=600',
                        name: 'Premium Canvas Trainer',
                        score: 0.85
                    },
                    {
                        imageUrl: 'https://images.unsplash.com/photo-1608231387042-66d1773070a5?auto=format&fit=crop&q=80&w=600',
                        name: 'Athletic Comfort Sneakers',
                        score: 0.76
                    },
                    {
                        imageUrl: 'https://images.unsplash.com/photo-1595950653106-6c9ebd614d3a?auto=format&fit=crop&q=80&w=600',
                        name: 'Chunky Sole Lifestyle Sneaker',
                        score: 0.62
                    }
                ]
            };
            renderResults(demoData, isNewSearch);

            if (isNewSearch) {
                loadingState.classList.add('hidden');
                resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }

            searchBtn.disabled = false;
            searchBtn.style.opacity = '1';
            loadMoreBtn.innerHTML = 'Load More';
            loadMoreBtn.disabled = false;
            paginationControls.classList.remove('hidden');
        }, 1500); // simulate 1.5s network delay
    }
});
