const courses = [
  {
    id: 1,
    title: "Introduction to HTML",
    description: "Learn the basics of HTML, the standard markup language for creating web pages.",
    lessons: [
      "What is HTML?",
      "Basic tags: html, head, body",
      "Common elements: headings, paragraphs, links"
    ]
  },
  {
    id: 2,
    title: "CSS Fundamentals",
    description: "Understand the fundamentals of CSS and how to style your web pages.",
    lessons: [
      "Selectors & Specificity",
      "Box Model & Layout",
      "Colors, Typography & Responsive design"
    ]
  },
  {
    id: 3,
    title: "JavaScript Basics",
    description: "Get started with JavaScript, the programming language of the web.",
    lessons: [
      "Variables & Types",
      "Functions & Control Flow",
      "DOM Manipulation basics"
    ]
  }
];

// localStorage helpers
function loadJSON(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch {
    return fallback;
  }
}
function saveJSON(key, value) {
  localStorage.setItem(key, JSON.stringify(value));
}

const STORAGE_COMPLETED = "completedCourses";
const STORAGE_PROGRESS = "courseProgress";

function getCompletedCourses() {
  return loadJSON(STORAGE_COMPLETED, []);
}
function saveCompletedCourses(list) {
  saveJSON(STORAGE_COMPLETED, list);
}
function getCourseProgress() {
  return loadJSON(STORAGE_PROGRESS, {}); // { [id]: [lessonIndex,...] }
}
function saveCourseProgress(obj) {
  saveJSON(STORAGE_PROGRESS, obj);
}

// helpers
function qs(sel){ return document.querySelector(sel); }
function qsa(sel){ return Array.from(document.querySelectorAll(sel)); }
function getQueryParam(name){
  const params = new URLSearchParams(location.search);
  return params.get(name);
}

// Render index: course list
function renderCourseList() {
  const section = qs("#course-list");
  if (!section) return;
  section.innerHTML = ""; // clear
  const completed = getCompletedCourses();
  courses.forEach(c => {
    const card = document.createElement("div");
    card.className = "course-card";
    card.dataset.courseId = c.id;
    card.innerHTML = `
      <h3>${escapeHtml(c.title)}</h3>
      <p>${escapeHtml(c.description)}</p>
    `;
    const actions = document.createElement("div");
    const link = document.createElement("a");
    link.href = `course.html?id=${c.id}`;
    link.textContent = "View Course";
    actions.appendChild(link);

    const btn = document.createElement("button");
    btn.textContent = completed.includes(c.id) ? "Completed" : "Mark Completed";
    btn.style.marginLeft = "8px";
    btn.disabled = completed.includes(c.id);
    btn.addEventListener("click", (e) => {
      e.preventDefault();
      toggleCourseComplete(c.id, true);
      btn.textContent = "Completed";
      btn.disabled = true;
      renderCourseList();
    });
    actions.appendChild(btn);

    card.appendChild(actions);
    section.appendChild(card);
  });
}

// Render course detail
function renderCourseDetail(courseId) {
  const course = courses.find(c => c.id === courseId);
  const titleEl = qs("#course-title");
  if (!course || !titleEl) return;
  titleEl.textContent = course.title;
  qs("#course-description").textContent = course.description;

  const lessonList = qs("#lesson-list");
  const progressBar = qs("#progress-bar");
  const progressText = qs("#progress-text");
  const completeBtn = qs("#complete-course-btn");

  // load progress data
  const progress = getCourseProgress();
  const completedLessons = new Set(progress[courseId] || []);
  const completedCourses = getCompletedCourses();
  const isCourseCompleted = completedCourses.includes(courseId);

  // render lessons
  lessonList.innerHTML = "";
  course.lessons.forEach((lesson, idx) => {
    const li = document.createElement("li");
    li.dataset.idx = idx;
    li.className = completedLessons.has(idx) ? "lesson-completed" : "";
    li.innerHTML = `
      <div>
        <div class="lesson-title">${escapeHtml(lesson)}</div>
        <div class="lesson-meta">${completedLessons.has(idx) ? "Completed" : "Not completed"}</div>
      </div>
      <div class="lesson-action">${completedLessons.has(idx) ? "✓" : "○"}</div>
    `;
    li.addEventListener("click", () => {
      toggleLesson(courseId, idx);
    });
    lessonList.appendChild(li);
  });

  // progress calculation
  updateProgressUI(courseId);

  // complete course button
  completeBtn.disabled = isCourseCompleted;
  completeBtn.textContent = isCourseCompleted ? "Completed" : "Mark as Completed";
  completeBtn.onclick = () => {
    toggleCourseComplete(courseId, !isCourseCompleted);
    updateProgressUI(courseId);
    completeBtn.disabled = true;
    completeBtn.textContent = "Completed";
    renderCourseList();
  };
}

function updateProgressUI(courseId) {
  const course = courses.find(c => c.id === courseId);
  if (!course) return;
  const progressBar = qs("#progress-bar");
  const progressText = qs("#progress-text");
  const progress = getCourseProgress();
  const completedLessons = (progress[courseId] || []).length;
  const percent = Math.round((completedLessons / course.lessons.length) * 100);
  if (progressBar) progressBar.value = percent;
  if (progressText) progressText.textContent = `${percent}%`;
}

// toggles a lesson completed state
function toggleLesson(courseId, lessonIdx) {
  const progress = getCourseProgress();
  const arr = new Set(progress[courseId] || []);
  if (arr.has(lessonIdx)) {
    arr.delete(lessonIdx);
  } else {
    arr.add(lessonIdx);
  }
  progress[courseId] = Array.from(arr).sort((a,b)=>a-b);
  saveCourseProgress(progress);
  renderCourseDetail(courseId);
}

// toggle course completed (if mark true, add it)
function toggleCourseComplete(courseId, markCompleted) {
  const completed = new Set(getCompletedCourses());
  if (markCompleted) {
    completed.add(courseId);
  } else {
    completed.delete(courseId);
  }
  saveCompletedCourses(Array.from(completed));
}

// small safe-escape for text nodes
function escapeHtml(str){
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

// init on DOM ready
document.addEventListener("DOMContentLoaded", () => {
  if (qs("#course-list")) {
    renderCourseList();
  }
  // course page
  if (qs("#course-title")) {
    const idParam = parseInt(getQueryParam("id"), 10);
    const courseId = Number.isInteger(idParam) ? idParam : null;
    if (!courseId) {
      qs("#course-title").textContent = "Course not found";
      return;
    }
    renderCourseDetail(courseId);
  }
});