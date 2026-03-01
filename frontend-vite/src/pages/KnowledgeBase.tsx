import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  IconBrain,
  IconPlus,
  IconTrash,
  IconSearch,
  IconLoader2,
  IconCheck,
  IconX,
  IconBook,
  IconMessageCircle
} from '@tabler/icons-react';
import api from '../utils/api';
import { useThemeStore } from '@store/themeStore';
import clsx from 'clsx';

interface RagNote {
  title?: string;
  text: string;
  metadata?: Record<string, any>;
}

interface RagFaq {
  question: string;
  answer: string;
}

export default function KnowledgeBase() {
  const { theme } = useThemeStore();
  const [activeTab, setActiveTab] = useState<'notes' | 'faqs'>('notes');
  
  // Notes states
  const [notes, setNotes] = useState<RagNote[]>([]);
  const [newNoteTitle, setNewNoteTitle] = useState('');
  const [newNoteText, setNewNoteText] = useState('');
  const [noteStatus, setNoteStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [noteError, setNoteError] = useState('');
  
  // FAQ states
  const [newQuestion, setNewQuestion] = useState('');
  const [newAnswer, setNewAnswer] = useState('');
  const [faqStatus, setFaqStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [faqError, setFaqError] = useState('');
  
  // Search
  const [searchTerm, setSearchTerm] = useState('');

  useEffect(() => {
    loadNotes();
  }, []);

  const loadNotes = async () => {
    try {
      const data = await api.listRagNotes();
      setNotes(data.notes || []);
    } catch (err) {
      console.error('Failed to load notes:', err);
    }
  };

  const addNote = async () => {
    if (!newNoteText.trim()) {
      setNoteError('Please enter note content');
      return;
    }

    try {
      setNoteStatus('loading');
      setNoteError('');
      
      await api.addRagNote({
        title: newNoteTitle.trim() || undefined,
        text: newNoteText.trim(),
        metadata: { source: 'kb-ui', timestamp: Date.now() }
      });

      setNoteStatus('success');
      setNewNoteTitle('');
      setNewNoteText('');
      await loadNotes();
      
      setTimeout(() => setNoteStatus('idle'), 2000);
    } catch (err: any) {
      console.error('Failed to add note:', err);
      setNoteStatus('error');
      setNoteError(err.message || 'Failed to add note');
    }
  };

  const addFaq = async () => {
    if (!newQuestion.trim() || !newAnswer.trim()) {
      setFaqError('Please enter both question and answer');
      return;
    }

    try {
      setFaqStatus('loading');
      setFaqError('');
      
      await api.addRagFaq({
        question: newQuestion.trim(),
        answer: newAnswer.trim()
      });

      setFaqStatus('success');
      setNewQuestion('');
      setNewAnswer('');
      
      setTimeout(() => setFaqStatus('idle'), 2000);
    } catch (err: any) {
      console.error('Failed to add FAQ:', err);
      setFaqStatus('error');
      setFaqError(err.message || 'Failed to add FAQ');
    }
  };

  const filteredNotes = notes.filter(note => 
    searchTerm === '' || 
    (note.title?.toLowerCase().includes(searchTerm.toLowerCase())) ||
    note.text.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className={clsx(
      'p-6 min-h-screen',
      theme === 'dark' ? 'bg-gray-900' : 'bg-gray-50'
    )}>
      <div className="max-w-6xl mx-auto space-y-6">
        {/* Header */}
        <div className={clsx(
          'p-6 rounded-xl border',
          theme === 'dark' ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'
        )}>
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-purple-500 to-blue-600 flex items-center justify-center">
              <IconBrain className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className={clsx(
                'text-3xl font-bold',
                theme === 'dark' ? 'text-white' : 'text-gray-900'
              )}>
                Knowledge Base
              </h1>
              <p className={clsx(
                'mt-1',
                theme === 'dark' ? 'text-gray-400' : 'text-gray-600'
              )}>
                Manage knowledge for the Voice Agent (RAG)
              </p>
            </div>
          </div>
        </div>

        {/* Search */}
        <div className={clsx(
          'p-4 rounded-xl border',
          theme === 'dark' ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'
        )}>
          <div className="relative">
            <IconSearch className={clsx(
              'absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5',
              theme === 'dark' ? 'text-gray-400' : 'text-gray-500'
            )} />
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Search knowledge base..."
              className={clsx(
                'w-full pl-10 pr-4 py-2.5 rounded-lg border focus:outline-none focus:ring-2 focus:ring-blue-500',
                theme === 'dark'
                  ? 'bg-gray-700 border-gray-600 text-white placeholder-gray-400'
                  : 'bg-white border-gray-300 text-gray-900 placeholder-gray-500'
              )}
            />
          </div>
        </div>

        {/* Tabs */}
        <div className={clsx(
          'flex gap-2 p-1 rounded-lg',
          theme === 'dark' ? 'bg-gray-800' : 'bg-gray-200'
        )}>
          <button
            onClick={() => setActiveTab('notes')}
            className={clsx(
              'flex-1 px-4 py-2.5 rounded-lg font-medium transition-colors flex items-center justify-center gap-2',
              activeTab === 'notes'
                ? 'bg-blue-600 text-white'
                : theme === 'dark'
                ? 'text-gray-400 hover:text-gray-200'
                : 'text-gray-600 hover:text-gray-900'
            )}
          >
            <IconBook size={18} />
            Notes & Policies
          </button>
          <button
            onClick={() => setActiveTab('faqs')}
            className={clsx(
              'flex-1 px-4 py-2.5 rounded-lg font-medium transition-colors flex items-center justify-center gap-2',
              activeTab === 'faqs'
                ? 'bg-blue-600 text-white'
                : theme === 'dark'
                ? 'text-gray-400 hover:text-gray-200'
                : 'text-gray-600 hover:text-gray-900'
            )}
          >
            <IconMessageCircle size={18} />
            FAQs
          </button>
        </div>

        {/* Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Add Form */}
          <div className="lg:col-span-1">
            <div className={clsx(
              'p-6 rounded-xl border sticky top-6',
              theme === 'dark' ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'
            )}>
              <h3 className={clsx(
                'text-lg font-semibold mb-4 flex items-center gap-2',
                theme === 'dark' ? 'text-white' : 'text-gray-900'
              )}>
                <IconPlus size={20} />
                {activeTab === 'notes' ? 'Add Note' : 'Add FAQ'}
              </h3>

              {activeTab === 'notes' ? (
                <div className="space-y-4">
                  <div>
                    <label className={clsx(
                      'block text-sm font-medium mb-2',
                      theme === 'dark' ? 'text-gray-300' : 'text-gray-700'
                    )}>
                      Title (optional)
                    </label>
                    <input
                      type="text"
                      value={newNoteTitle}
                      onChange={(e) => setNewNoteTitle(e.target.value)}
                      placeholder="e.g., Clinic Hours Policy"
                      className={clsx(
                        'w-full px-3 py-2 rounded-lg border focus:outline-none focus:ring-2 focus:ring-blue-500',
                        theme === 'dark'
                          ? 'bg-gray-700 border-gray-600 text-white placeholder-gray-400'
                          : 'bg-white border-gray-300 text-gray-900 placeholder-gray-500'
                      )}
                    />
                  </div>

                  <div>
                    <label className={clsx(
                      'block text-sm font-medium mb-2',
                      theme === 'dark' ? 'text-gray-300' : 'text-gray-700'
                    )}>
                      Content
                    </label>
                    <textarea
                      value={newNoteText}
                      onChange={(e) => setNewNoteText(e.target.value)}
                      placeholder="Enter clinic policies, guidelines, medication tables, treatment protocols, etc..."
                      rows={8}
                      className={clsx(
                        'w-full px-3 py-2 rounded-lg border focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none',
                        theme === 'dark'
                          ? 'bg-gray-700 border-gray-600 text-white placeholder-gray-400'
                          : 'bg-white border-gray-300 text-gray-900 placeholder-gray-500'
                      )}
                    />
                  </div>

                  <button
                    onClick={addNote}
                    disabled={noteStatus === 'loading'}
                    className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-medium py-2.5 px-4 rounded-lg transition-colors flex items-center justify-center gap-2"
                  >
                    {noteStatus === 'loading' ? (
                      <>
                        <IconLoader2 className="w-5 h-5 animate-spin" />
                        Adding...
                      </>
                    ) : noteStatus === 'success' ? (
                      <>
                        <IconCheck className="w-5 h-5" />
                        Added!
                      </>
                    ) : (
                      <>
                        <IconPlus className="w-5 h-5" />
                        Add Note
                      </>
                    )}
                  </button>

                  {noteError && (
                    <div className={clsx(
                      'p-3 rounded-lg flex items-start gap-2',
                      theme === 'dark' ? 'bg-red-500/20 border border-red-500/30' : 'bg-red-50 border border-red-200'
                    )}>
                      <IconX className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
                      <p className="text-sm text-red-500">{noteError}</p>
                    </div>
                  )}
                </div>
              ) : (
                <div className="space-y-4">
                  <div>
                    <label className={clsx(
                      'block text-sm font-medium mb-2',
                      theme === 'dark' ? 'text-gray-300' : 'text-gray-700'
                    )}>
                      Question
                    </label>
                    <textarea
                      value={newQuestion}
                      onChange={(e) => setNewQuestion(e.target.value)}
                      placeholder="What question might patients ask?"
                      rows={3}
                      className={clsx(
                        'w-full px-3 py-2 rounded-lg border focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none',
                        theme === 'dark'
                          ? 'bg-gray-700 border-gray-600 text-white placeholder-gray-400'
                          : 'bg-white border-gray-300 text-gray-900 placeholder-gray-500'
                      )}
                    />
                  </div>

                  <div>
                    <label className={clsx(
                      'block text-sm font-medium mb-2',
                      theme === 'dark' ? 'text-gray-300' : 'text-gray-700'
                    )}>
                      Answer
                    </label>
                    <textarea
                      value={newAnswer}
                      onChange={(e) => setNewAnswer(e.target.value)}
                      placeholder="The answer to provide..."
                      rows={3}
                      className={clsx(
                        'w-full px-3 py-2 rounded-lg border focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none',
                        theme === 'dark'
                          ? 'bg-gray-700 border-gray-600 text-white placeholder-gray-400'
                          : 'bg-white border-gray-300 text-gray-900 placeholder-gray-500'
                      )}
                    />
                  </div>

                  <button
                    onClick={addFaq}
                    disabled={faqStatus === 'loading'}
                    className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-medium py-2.5 px-4 rounded-lg transition-colors flex items-center justify-center gap-2"
                  >
                    {faqStatus === 'loading' ? (
                      <>
                        <IconLoader2 className="w-5 h-5 animate-spin" />
                        Adding...
                      </>
                    ) : faqStatus === 'success' ? (
                      <>
                        <IconCheck className="w-5 h-5" />
                        Added!
                      </>
                    ) : (
                      <>
                        <IconPlus className="w-5 h-5" />
                        Add FAQ
                      </>
                    )}
                  </button>

                  {faqError && (
                    <div className={clsx(
                      'p-3 rounded-lg flex items-start gap-2',
                      theme === 'dark' ? 'bg-red-500/20 border border-red-500/30' : 'bg-red-50 border border-red-200'
                    )}>
                      <IconX className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
                      <p className="text-sm text-red-500">{faqError}</p>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Knowledge List */}
          <div className="lg:col-span-2">
            <div className={clsx(
              'p-6 rounded-xl border',
              theme === 'dark' ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'
            )}>
              <h3 className={clsx(
                'text-lg font-semibold mb-4',
                theme === 'dark' ? 'text-white' : 'text-gray-900'
              )}>
                {activeTab === 'notes' ? 'Knowledge Notes' : 'FAQ List'}
              </h3>

              {activeTab === 'notes' && (
                <div className="space-y-4 max-h-[700px] overflow-y-auto pr-2">
                  {filteredNotes.length === 0 ? (
                    <div className="text-center py-12">
                      <IconBook className={clsx(
                        'w-16 h-16 mx-auto mb-4',
                        theme === 'dark' ? 'text-gray-600' : 'text-gray-300'
                      )} />
                      <p className={clsx(
                        theme === 'dark' ? 'text-gray-400' : 'text-gray-500'
                      )}>
                        {searchTerm ? 'No notes match your search' : 'No notes yet. Add your first knowledge note!'}
                      </p>
                    </div>
                  ) : (
                    filteredNotes.map((note, index) => (
                      <motion.div
                        key={index}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: index * 0.05 }}
                        className={clsx(
                          'p-4 rounded-lg border',
                          theme === 'dark' ? 'bg-gray-700/50 border-gray-600' : 'bg-gray-50 border-gray-200'
                        )}
                      >
                        {note.title && (
                          <h4 className={clsx(
                            'font-semibold mb-2',
                            theme === 'dark' ? 'text-white' : 'text-gray-900'
                          )}>
                            {note.title}
                          </h4>
                        )}
                        <p className={clsx(
                          'text-sm whitespace-pre-wrap',
                          theme === 'dark' ? 'text-gray-300' : 'text-gray-700'
                        )}>
                          {note.text}
                        </p>
                        {note.metadata && (
                          <div className={clsx(
                            'mt-3 pt-3 border-t text-xs',
                            theme === 'dark' ? 'border-gray-600 text-gray-500' : 'border-gray-200 text-gray-500'
                          )}>
                            Added: {note.metadata.timestamp ? new Date(note.metadata.timestamp).toLocaleString() : 'Unknown'}
                          </div>
                        )}
                      </motion.div>
                    ))
                  )}
                </div>
              )}

              {activeTab === 'faqs' && (
                <div className="text-center py-12">
                  <IconMessageCircle className={clsx(
                    'w-16 h-16 mx-auto mb-4',
                    theme === 'dark' ? 'text-gray-600' : 'text-gray-300'
                  )} />
                  <p className={clsx(
                    theme === 'dark' ? 'text-gray-400' : 'text-gray-500'
                  )}>
                    FAQs can be added and will be available to the Voice Agent
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Info Panel */}
        <div className={clsx(
          'p-6 rounded-xl border',
          theme === 'dark' ? 'bg-blue-500/10 border-blue-500/30' : 'bg-blue-50 border-blue-200'
        )}>
          <div className="flex gap-4">
            <div className={clsx(
              'w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0',
              theme === 'dark' ? 'bg-blue-500/20' : 'bg-blue-100'
            )}>
              <IconBrain className="w-6 h-6 text-blue-500" />
            </div>
            <div>
              <h4 className={clsx(
                'font-semibold mb-1',
                theme === 'dark' ? 'text-blue-300' : 'text-blue-900'
              )}>
                How it works
              </h4>
              <p className={clsx(
                'text-sm',
                theme === 'dark' ? 'text-blue-200/80' : 'text-blue-800/80'
              )}>
                Knowledge added here will be used by the Voice Agent (VA) to provide accurate,
                context-aware responses. Add clinic policies, treatment protocols, medication information,
                frequently asked questions, and any other knowledge your VA should have access to.
                The system uses RAG (Retrieval-Augmented Generation) to find relevant knowledge when needed.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
