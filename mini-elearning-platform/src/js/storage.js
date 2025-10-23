// This file manages the storage of course completion status, possibly using local storage to persist user progress.

const storageKey = 'courseCompletionStatus';

// Function to get the completion status of a course
function getCourseCompletionStatus(courseId) {
    const status = JSON.parse(localStorage.getItem(storageKey)) || {};
    return status[courseId] || false;
}

// Function to set the completion status of a course
function setCourseCompletionStatus(courseId, isCompleted) {
    const status = JSON.parse(localStorage.getItem(storageKey)) || {};
    status[courseId] = isCompleted;
    localStorage.setItem(storageKey, JSON.stringify(status));
}

// Function to clear all completion statuses (for testing purposes)
function clearCompletionStatus() {
    localStorage.removeItem(storageKey);
}

// Exporting the functions for use in other modules
export { getCourseCompletionStatus, setCourseCompletionStatus, clearCompletionStatus };