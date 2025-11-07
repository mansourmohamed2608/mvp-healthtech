// gateway/src/queue/queue.service.ts
import { Injectable, Logger } from '@nestjs/common';

interface QueueItem<T> {
  id: string;
  sessionId: string;
  data: T;
  priority: number; // 0 = highest, 10 = lowest
  timestamp: number;
  resolve: (value: any) => void;
  reject: (error: any) => void;
}

interface QueueMetrics {
  size: number;
  processing: number;
  completed: number;
  failed: number;
  avgProcessingTime: number;
  maxProcessingTime: number;
}

@Injectable()
export class QueueService {
  private readonly logger = new Logger(QueueService.name);
  
  // Queue configuration
  private readonly MAX_QUEUE_SIZE = 100;
  private readonly MAX_CONCURRENT = 10;
  private readonly MAX_PER_SESSION = 5; // Max concurrent requests per session
  private readonly REQUEST_TIMEOUT = 30000; // 30 seconds

  // Queue state
  private readonly queue: QueueItem<any>[] = [];
  private readonly processing = new Map<string, QueueItem<any>>();
  private readonly sessionCounts = new Map<string, number>();
  
  // Metrics
  private completed = 0;
  private failed = 0;
  private processingTimes: number[] = [];
  private readonly MAX_TIMES_STORED = 100;

  /**
   * Add item to queue with back-pressure control
   */
  async enqueue<T, R>(
    sessionId: string,
    data: T,
    processor: (data: T) => Promise<R>,
    priority: number = 5,
  ): Promise<R> {
    // Check queue size limit
    if (this.queue.length >= this.MAX_QUEUE_SIZE) {
      throw new Error('Queue is full. System is overloaded.');
    }

    // Check per-session limit (back-pressure)
    const sessionCount = this.sessionCounts.get(sessionId) || 0;
    if (sessionCount >= this.MAX_PER_SESSION) {
      throw new Error(`Too many concurrent requests for session ${sessionId}. Please slow down.`);
    }

    // Create queue item with promise
    const item: QueueItem<T> = {
      id: `${sessionId}-${Date.now()}-${Math.random()}`,
      sessionId,
      data,
      priority,
      timestamp: Date.now(),
      resolve: null as any,
      reject: null as any,
    };

    // Create promise that will be resolved when processing completes
    const promise = new Promise<R>((resolve, reject) => {
      item.resolve = resolve;
      item.reject = reject;
    });

    // Add timeout
    const timeout = setTimeout(() => {
      this.removeFromQueue(item.id);
      item.reject(new Error('Request timeout'));
      this.failed++;
    }, this.REQUEST_TIMEOUT);

    // Store processor with item
    (item as any).processor = processor;
    (item as any).timeout = timeout;

    // Add to queue (sorted by priority)
    this.insertByPriority(item);
    
    // Increment session count
    this.sessionCounts.set(sessionId, sessionCount + 1);

    this.logger.debug(`Enqueued item ${item.id} for session ${sessionId} (priority: ${priority})`);

    // Try to process immediately
    this.processNext();

    return promise;
  }

  /**
   * Insert item into queue maintaining priority order
   */
  private insertByPriority(item: QueueItem<any>): void {
    let inserted = false;
    for (let i = 0; i < this.queue.length; i++) {
      if (item.priority < this.queue[i].priority) {
        this.queue.splice(i, 0, item);
        inserted = true;
        break;
      }
    }
    if (!inserted) {
      this.queue.push(item);
    }
  }

  /**
   * Process next item in queue if capacity available
   */
  private async processNext(): Promise<void> {
    // Check if we can process more
    if (this.processing.size >= this.MAX_CONCURRENT) {
      return;
    }

    // Get next item
    const item = this.queue.shift();
    if (!item) {
      return;
    }

    // Move to processing
    this.processing.set(item.id, item);

    const startTime = Date.now();

    try {
      // Execute processor
      const processor = (item as any).processor;
      const result = await processor(item.data);

      // Success
      const processingTime = Date.now() - startTime;
      this.recordProcessingTime(processingTime);
      
      clearTimeout((item as any).timeout);
      item.resolve(result);
      this.completed++;

      this.logger.debug(`Completed item ${item.id} in ${processingTime}ms`);
    } catch (error) {
      // Failure
      clearTimeout((item as any).timeout);
      item.reject(error);
      this.failed++;

      this.logger.error(`Failed to process item ${item.id}`, error);
    } finally {
      // Cleanup
      this.processing.delete(item.id);
      const sessionCount = this.sessionCounts.get(item.sessionId) || 0;
      this.sessionCounts.set(item.sessionId, Math.max(0, sessionCount - 1));

      // Process next item
      setImmediate(() => this.processNext());
    }
  }

  /**
   * Remove item from queue
   */
  private removeFromQueue(itemId: string): boolean {
    const index = this.queue.findIndex(item => item.id === itemId);
    if (index !== -1) {
      this.queue.splice(index, 1);
      return true;
    }
    return false;
  }

  /**
   * Record processing time for metrics
   */
  private recordProcessingTime(time: number): void {
    this.processingTimes.push(time);
    if (this.processingTimes.length > this.MAX_TIMES_STORED) {
      this.processingTimes.shift();
    }
  }

  /**
   * Get queue metrics
   */
  getMetrics(): QueueMetrics {
    const avgTime = this.processingTimes.length > 0
      ? this.processingTimes.reduce((sum, t) => sum + t, 0) / this.processingTimes.length
      : 0;
    
    const maxTime = this.processingTimes.length > 0
      ? Math.max(...this.processingTimes)
      : 0;

    return {
      size: this.queue.length,
      processing: this.processing.size,
      completed: this.completed,
      failed: this.failed,
      avgProcessingTime: Math.round(avgTime),
      maxProcessingTime: Math.round(maxTime),
    };
  }

  /**
   * Get session-specific metrics
   */
  getSessionMetrics(sessionId: string): {
    queuedItems: number;
    processingItems: number;
    allowedConcurrency: number;
    availableSlots: number;
  } {
    const queued = this.queue.filter(item => item.sessionId === sessionId).length;
    const processing = Array.from(this.processing.values())
      .filter(item => item.sessionId === sessionId).length;
    
    return {
      queuedItems: queued,
      processingItems: processing,
      allowedConcurrency: this.MAX_PER_SESSION,
      availableSlots: Math.max(0, this.MAX_PER_SESSION - processing),
    };
  }

  /**
   * Clear queue for a specific session (e.g., on call end)
   */
  clearSession(sessionId: string): number {
    let cleared = 0;

    // Remove from queue
    for (let i = this.queue.length - 1; i >= 0; i--) {
      if (this.queue[i].sessionId === sessionId) {
        const item = this.queue[i];
        clearTimeout((item as any).timeout);
        item.reject(new Error('Session cleared'));
        this.queue.splice(i, 1);
        cleared++;
      }
    }

    // Reset session count
    this.sessionCounts.delete(sessionId);

    this.logger.log(`Cleared ${cleared} items for session ${sessionId}`);
    return cleared;
  }

  /**
   * Health check
   */
  isHealthy(): boolean {
    return this.queue.length < this.MAX_QUEUE_SIZE * 0.9 && 
           this.processing.size <= this.MAX_CONCURRENT;
  }
}
