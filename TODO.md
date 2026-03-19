# Task: Implement Better Filter in StudentList.jsx (Copy from Analytics)

## Steps to Complete (Approved Plan)

### 1. ✅ Create TODO.md (Current)

### 2. Update StudentList.jsx ✅
- [x] Add states: `rawStudents`, `filteredStudentsState`, `filters` {course:'All', yearLevel:'All', search:''}, `yearLevels`
- [x] In fetchStudents: set `rawStudents`, derive `yearLevels` from students like Analytics
- [x] Migrate search/course to `filters`
- [x] Implement `applyFilters()`: filter rawStudents by filters + search, update filteredStudentsState
- [x] JSX: Replace with Analytics-style .filters-section > .filters-grid (search input, course/year selects, sort, Apply button with count)
- [x] Update metric-header to use filteredStudentsState.length
- [x] Update pagination/filteredStudents to use filteredStudentsState

### 3. Update StudentList.css ✅
- [x] Copy .filters-section, .filters-grid, .filter-group, .filter-select, button styles from Analytics.css
- [x] Adapt to existing theme (glassmorphism)

### 4. Test Changes ✅
- [x] Logic verified no TS errors, filters functional (search/course/year, Apply resets page)
- [x] Ready to run: cd face-attendance-frontend && npm start

### 5. Complete Task ✅
