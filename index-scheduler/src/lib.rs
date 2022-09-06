pub mod error;
pub mod task;

use error::Error;
use milli::heed::types::{DecodeIgnore, OwnedType, SerdeBincode, Str};
pub use task::Task;
use task::{Kind, KindWithContent, Status};

use std::collections::hash_map::Entry;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::{collections::HashMap, sync::RwLock};

use milli::heed::{Database, Env, EnvOpenOptions, RoTxn, RwTxn};
use milli::{Index, RoaringBitmapCodec, BEU32};
use roaring::RoaringBitmap;

pub type Result<T> = std::result::Result<T, Error>;
pub type TaskId = u32;
type IndexName = String;
type IndexUuid = String;

/// This module is responsible for two things;
/// 1. Resolve the name of the indexes.
/// 2. Schedule the tasks.

#[derive(Clone)]
pub struct IndexScheduler {
    // Keep track of the opened indexes and is used
    // mainly by the index resolver.
    index_map: Arc<RwLock<HashMap<IndexUuid, Index>>>,

    /// The list of tasks currently processing.
    processing_tasks: Arc<RwLock<RoaringBitmap>>,

    /// The LMDB environment which the DBs are associated with.
    env: Env,

    // The main database, it contains all the tasks accessible by their Id.
    all_tasks: Database<OwnedType<BEU32>, SerdeBincode<Task>>,

    // All the tasks ids grouped by their status.
    status: Database<SerdeBincode<Status>, RoaringBitmapCodec>,
    // All the tasks ids grouped by their kind.
    kind: Database<SerdeBincode<Kind>, RoaringBitmapCodec>,

    // Map an index name with an indexuuid.
    index_name_mapper: Database<Str, Str>,
    // Store the tasks associated to an index.
    index_tasks: Database<Str, RoaringBitmapCodec>,

    // set to true when there is work to do.
    wake_up: Arc<AtomicBool>,
}

impl IndexScheduler {
    pub fn index(&self, name: &str) -> Result<Index> {
        let rtxn = self.env.read_txn()?;
        let uuid = self
            .index_name_mapper
            .get(&rtxn, name)?
            .ok_or(Error::IndexNotFound)?;
        // we clone here to drop the lock before entering the match
        let index = self.index_map.read().unwrap().get(&*uuid).cloned();
        let index = match index {
            Some(index) => index,
            // since we're lazy, it's possible that the index doesn't exist yet.
            // We need to open it ourselves.
            None => {
                let mut index_map = self.index_map.write().unwrap();
                // between the read lock and the write lock it's not impossible
                // that someone already opened the index (eg if two search happens
                // at the same time), thus before opening it we check a second time
                // if it's not already there.
                // Since there is a good chance it's not already there we can use
                // the entry method.
                match index_map.entry(uuid.to_string()) {
                    Entry::Vacant(entry) => {
                        // TODO: TAMO: get the envopenoptions from somewhere
                        let index = milli::Index::new(EnvOpenOptions::new(), uuid)?;
                        entry.insert(index.clone());
                        index
                    }
                    Entry::Occupied(entry) => entry.get().clone(),
                }
            }
        };

        Ok(index)
    }

    fn next_task_id(&self, rtxn: &RoTxn) -> Result<TaskId> {
        Ok(self
            .all_tasks
            .remap_data_type::<DecodeIgnore>()
            .last(rtxn)?
            .map(|(k, _)| k.get())
            .unwrap_or(0))
    }

    /// Register a new task in the scheduler. If it fails and data was associated with the task
    /// it tries to delete the file.
    pub fn register(&self, task: Task) -> Result<()> {
        let mut wtxn = self.env.write_txn()?;

        let task_id = self.next_task_id(&wtxn)?;

        self.all_tasks
            .append(&mut wtxn, &BEU32::new(task_id), &task)?;

        self.update_status(&mut wtxn, Status::Enqueued, |mut bitmap| {
            bitmap.insert(task_id);
            bitmap
        })?;

        self.update_kind(&mut wtxn, task.kind.as_kind(), |mut bitmap| {
            bitmap.insert(task_id);
            bitmap
        })?;

        // we persist the file in last to be sure everything before was applied successfuly
        task.persist()?;

        match wtxn.commit() {
            Ok(()) => (),
            e @ Err(_) => {
                task.remove_data()?;
                e?;
            }
        }

        self.notify();

        Ok(())
    }

    /// Create the next batch to be processed;
    /// 1. We get the *last* task to cancel.
    /// 2. We get the *next* snapshot to process.
    /// 3. We get the *next* dump to process.
    /// 4. We get the *next* tasks to process for a specific index.
    fn get_next_batch(&self, rtxn: &RoTxn) -> Result<Batch> {
        let enqueued = &self.get_status(rtxn, Status::Enqueued)?;
        let to_cancel = self.get_kind(rtxn, Kind::CancelTask)? & enqueued;

        // 1. we get the last task to cancel.
        if let Some(task_id) = to_cancel.max() {
            return Ok(Batch::Cancel(
                self.get_task(rtxn, task_id)?
                    .ok_or(Error::CorruptedTaskQueue)?,
            ));
        }

        // 2. we batch the snapshot.
        let to_snapshot = self.get_kind(rtxn, Kind::Snapshot)? & enqueued;
        if !to_snapshot.is_empty() {
            return Ok(Batch::Snapshot(self.get_existing_tasks(rtxn, to_snapshot)?));
        }

        // 3. we batch the dumps.
        let to_dump = self.get_kind(rtxn, Kind::DumpExport)? & enqueued;
        if !to_dump.is_empty() {
            return Ok(Batch::Dump(self.get_existing_tasks(rtxn, to_dump)?));
        }

        // 4. We take the next task and try to batch all the tasks associated with this index.
        if let Some(task_id) = enqueued.min() {
            let task = self
                .get_task(rtxn, task_id)?
                .ok_or(Error::CorruptedTaskQueue)?;
            match task.kind {
                // We can batch all the consecutive tasks coming next which
                // have the kind `DocumentAddition`.
                KindWithContent::DocumentAddition { index_name, .. } => {
                    return self.batch_contiguous_kind(rtxn, &index_name, Kind::DocumentAddition)
                }
                // We can batch all the consecutive tasks coming next which
                // have the kind `DocumentDeletion`.
                KindWithContent::DocumentDeletion { index_name, .. } => {
                    return self.batch_contiguous_kind(rtxn, &index_name, Kind::DocumentAddition)
                }
                // The following tasks can't be batched
                KindWithContent::ClearAllDocuments { .. }
                | KindWithContent::RenameIndex { .. }
                | KindWithContent::CreateIndex { .. }
                | KindWithContent::DeleteIndex { .. }
                | KindWithContent::SwapIndex { .. } => return Ok(Batch::One(task)),

                // The following tasks have already been batched and thus can't appear here.
                KindWithContent::CancelTask { .. }
                | KindWithContent::DumpExport { .. }
                | KindWithContent::Snapshot => {
                    unreachable!()
                }
            }
        }

        // If we found no tasks then we were notified for something that got autobatched
        // somehow and there is nothing to do.
        Ok(Batch::Empty)
    }

    /// Batch all the consecutive tasks coming next that shares the same `Kind`
    /// for a specific index. There *MUST* be at least ONE task of this kind.
    fn batch_contiguous_kind(&self, rtxn: &RoTxn, index: &str, kind: Kind) -> Result<Batch> {
        let enqueued = &self.get_status(rtxn, Status::Enqueued)?;

        // [1, 2, 4, 5]
        let index_tasks = self.get_index(rtxn, &index)? & enqueued;
        // [1, 2, 5]
        let tasks_kind = &index_tasks & self.get_kind(rtxn, kind)?;
        // [4]
        let not_kind = &index_tasks - &tasks_kind;

        // [1, 2]
        let mut to_process = tasks_kind.clone();
        if let Some(max) = not_kind.max() {
            // it's safe to unwrap since we already ensured there
            // was AT LEAST one task with the document addition tasks_kind.
            to_process.remove_range(tasks_kind.min().unwrap()..max);
        }

        Ok(Batch::Contiguous {
            tasks: self.get_existing_tasks(rtxn, to_process)?,
            kind,
        })
    }

    fn get_task(&self, rtxn: &RoTxn, task_id: TaskId) -> Result<Option<Task>> {
        Ok(self.all_tasks.get(rtxn, &BEU32::new(task_id))?)
    }

    pub fn notify(&self) {
        self.wake_up
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    // =========== Utility functions on the DBs

    /// Convert an iterator to a `Vec` of tasks. The tasks MUST exist or a
    // `CorruptedTaskQueue` error will be throwed.
    fn get_existing_tasks(
        &self,
        rtxn: &RoTxn,
        tasks: impl IntoIterator<Item = TaskId>,
    ) -> Result<Vec<Task>> {
        tasks
            .into_iter()
            .map(|task_id| {
                self.get_task(rtxn, task_id)
                    .and_then(|task| task.ok_or(Error::CorruptedTaskQueue))
            })
            .collect::<Result<_>>()
    }

    fn get_index(&self, rtxn: &RoTxn, index: &str) -> Result<RoaringBitmap> {
        Ok(self.index_tasks.get(&rtxn, index)?.unwrap_or_default())
    }

    fn put_index(&self, wtxn: &mut RwTxn, index: &str, bitmap: &RoaringBitmap) -> Result<()> {
        Ok(self.index_tasks.put(wtxn, index, bitmap)?)
    }

    fn get_status(&self, rtxn: &RoTxn, status: Status) -> Result<RoaringBitmap> {
        Ok(self.status.get(&rtxn, &status)?.unwrap_or_default())
    }

    fn put_status(&self, wtxn: &mut RwTxn, status: Status, bitmap: &RoaringBitmap) -> Result<()> {
        Ok(self.status.put(wtxn, &status, bitmap)?)
    }

    fn get_kind(&self, rtxn: &RoTxn, kind: Kind) -> Result<RoaringBitmap> {
        Ok(self.kind.get(&rtxn, &kind)?.unwrap_or_default())
    }

    fn put_kind(&self, wtxn: &mut RwTxn, kind: Kind, bitmap: &RoaringBitmap) -> Result<()> {
        Ok(self.kind.put(wtxn, &kind, bitmap)?)
    }

    fn update_status(
        &self,
        wtxn: &mut RwTxn,
        status: Status,
        f: impl Fn(RoaringBitmap) -> RoaringBitmap,
    ) -> Result<()> {
        let tasks = self.get_status(&wtxn, status)?;
        let tasks = f(tasks);
        self.put_status(wtxn, status, &tasks)?;

        Ok(())
    }

    fn update_kind(
        &self,
        wtxn: &mut RwTxn,
        kind: Kind,
        f: impl Fn(RoaringBitmap) -> RoaringBitmap,
    ) -> Result<()> {
        let tasks = self.get_kind(&wtxn, kind)?;
        let tasks = f(tasks);
        self.put_kind(wtxn, kind, &tasks)?;

        Ok(())
    }
}

enum Batch {
    Cancel(Task),
    Snapshot(Vec<Task>),
    Dump(Vec<Task>),
    Contiguous { tasks: Vec<Task>, kind: Kind },
    One(Task),
    Empty,
}
